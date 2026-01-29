import numpy as np
from services.encode_hkl import encode_hkl
import copy
import plotly.graph_objs as go

import logging


class ScanOptimizer:
    def __init__(self, experiment):
        self.exp = experiment
        self.target_scan_index = None

        self.total_refs_count = 0
        self.encoded_to_dense_map = {}
        self.base_counts = None
        self.sim_angles = None
        self.sim_dense_indices = None
        self.sim_hkl_indices = None


        self.base_runs = set()




        self._is_initialized = False

        self._original_scan_params = None
        self._is_modified = False

        self._maps_data_cache = {}
        self._plotter = None

    def set_plotter(self, plotter):
        """Связывает оптимизатор с плоттером для синхронной очистки кэша."""
        self._plotter = plotter

    def select_scan(self, scan_index):
        self._maps_data_cache.clear()
        if self._plotter:
            self._plotter.clear_cache()

        self.target_scan_index = scan_index
        current_scan = self.exp.scans[scan_index]
        self._original_scan_params = copy.deepcopy(current_scan)
        self._is_modified = False




        if not self._is_initialized:

            if self.exp.scans:
                self.base_runs = set(range(len(self.exp.scans)))
            self._is_initialized = True




        self._precalculate_data()

    def exclude_base_run(self, scan_index):
        """Исключает скан из расчета фона."""
        if scan_index in self.base_runs:
            self.base_runs.remove(scan_index)

            self._precalculate_data()
            self._maps_data_cache.clear()
            self._clear_plotter()

    def include_base_run(self, scan_index):
        """Возвращает скан в расчет фона."""
        if scan_index not in self.base_runs:
            self.base_runs.add(scan_index)

            self._precalculate_data()
            self._maps_data_cache.clear()
            self._clear_plotter()

    def apply_parameters(self, new_start_angle, new_sweep):
        """
        Применяет новые параметры к эксперименту и обновляет данные в StrategyContainer.
        Это 'Preview' режим - данные обновлены, но можно откатиться.
        """
        if self.target_scan_index is None:
            raise ValueError("Scan not selected")

        scan_idx = self.target_scan_index
        scan_data = list(self.exp.scans[scan_idx])
        axis_index = scan_data[4]

        new_axes_angles = list(scan_data[3])
        new_axes_angles[axis_index] = float(new_start_angle)
        scan_data[3] = new_axes_angles
        scan_data[5] = float(new_sweep)

        self.exp.scans[scan_idx] = tuple(scan_data)
        self._is_modified = True

        self._recalculate_single_scan_container(scan_idx)

        self._maps_data_cache.clear()

        self._clear_plotter()

    def _clear_plotter(self):
        if self._plotter:
            self._plotter.clear_cache()

    def revert_changes(self):
        if not self._is_modified or self.target_scan_index is None:
            return

        self.exp.scans[self.target_scan_index] = self._original_scan_params
        self._recalculate_single_scan_container(self.target_scan_index)
        self._is_modified = False

        self._maps_data_cache.clear()
        self._clear_plotter()

    def _recalculate_single_scan_container(self, scan_idx):
        """
        Внутренний метод: вызывает cell.scan, применяет фильтры
        и обновляет StrategyContainer.
        """
        scan = self.exp.scans[scan_idx]

        hkl_in = self.exp.hkl_in_d_range
        hkl_origin_in = self.exp.hkl_origin_in_d_range

        data = self.exp.cell.scan(
            no_of_scan=scan[4],
            scan_sweep=scan[5],
            wavelength=self.exp.wavelength,
            angles=scan[3],
            hkl_array=hkl_in,
            hkl_array_orig=hkl_origin_in,
            directions=self.exp.axes_directions,
            rotations=self.exp.axes_rotations
        )

        if self.exp.diamond_anvil and self.exp.calc_anvil_flag:
            data = self.exp.diamond_anvil.filter_anvil(
                diff_vecs=data[0], diff_angles=data[3],
                rotation_axes=self.exp.axes_rotations,
                directions_axes=self.exp.axes_directions,
                initial_axes_angles=scan[3],
                scan_axis_index=scan[4],
                data=data, mode='transmit',
                incident_beam=self.exp.incident_beam_vec
            )

        if self.exp.det_geometry is not None:
            data = self.exp._apply_detector(
                data=data, dist=scan[0], orientation=scan[2],
                rot=scan[1], disp_y=scan[6], disp_z=scan[7]
            )

        for lo in self.exp.linked_obstacles:
            data = lo.filter_linked_obstacle(
                scan_axis_index=scan[4], diff_vectors=data[0],
                initial_axes_angles=scan[3], diff_angles=data[3], mode='shade',
                directions_axes=self.exp.axes_directions,
                rotation_axes=self.exp.axes_rotations, data=data
            )

        if hasattr(self.exp, 'obstacles') and self.exp.obstacles:
            for obstacle in self.exp.obstacles:
                obstacle_ = self.exp._create_obstacle(obstacle)
                data = obstacle_.filter(diff_vecs=data[0], data=data, mode='shade')

        import services.service_functions as sf

        new_sdc = sf.ScanDataContainer(
            scan=scan,
            diff_vecs=data[0],
            hkl=data[1],
            hkl_origin=data[2],
            diff_angles=data[3],
            scan_setup=scan,
            start_angle=scan[3][scan[4]],
            sweep=scan[5]
        )

        self.exp.strategy_data_container.scan_data_containers[scan_idx] = new_sdc

    def _precalculate_data(self):
        if self.exp.hkl_origin_in_d_range is None:
            raise ValueError("Сначала запустите experiment.calc_experiment()")

        raw_hkl_origin = self.exp.hkl_origin_in_d_range
        unique_hkl_refs = np.unique(raw_hkl_origin, axis=0)
        self.total_unique_refs_count = len(unique_hkl_refs)

        encoded_unique = encode_hkl(unique_hkl_refs.astype(int)).flatten()
        self.encoded_to_dense_map = {code: i for i, code in enumerate(encoded_unique)}

        self.base_counts = np.zeros(self.total_unique_refs_count, dtype=np.int32)

        for i, sdc in enumerate(self.exp.strategy_data_container):
            if i == self.target_scan_index:
                continue

            if i not in self.base_runs:
                continue

            hkl_orig = sdc.hkl_origin.astype(int)
            encoded_current = encode_hkl(hkl_orig).flatten()
            indices = [self.encoded_to_dense_map.get(code) for code in encoded_current]
            indices = [x for x in indices if x is not None]

            if indices:
                np.add.at(self.base_counts, indices, 1)

        target_scan = self.exp.scans[self.target_scan_index]
        scan_axis_index = target_scan[4]
        initial_angles = list(target_scan[3])
        initial_angles[scan_axis_index] = 0.0

        hkl_in = self.exp.hkl_in_d_range
        hkl_origin_in = self.exp.hkl_origin_in_d_range

        data = self.exp.cell.scan(
            no_of_scan=scan_axis_index,
            scan_sweep=360.0,
            wavelength=self.exp.wavelength,
            angles=initial_angles,
            hkl_array=hkl_in,
            hkl_array_orig=hkl_origin_in,
            directions=self.exp.axes_directions,
            rotations=self.exp.axes_rotations
        )


        if self.exp.diamond_anvil and self.exp.calc_anvil_flag:
            data = self.exp.diamond_anvil.filter_anvil(
                diff_vecs=data[0], diff_angles=data[3],
                rotation_axes=self.exp.axes_rotations,
                directions_axes=self.exp.axes_directions,
                initial_axes_angles=initial_angles,
                scan_axis_index=scan_axis_index,
                data=data, mode='transmit', incident_beam=self.exp.incident_beam_vec
            )

        if self.exp.det_geometry is not None:
            data = self.exp._apply_detector(
                data=data, dist=target_scan[0], orientation=target_scan[2],
                rot=target_scan[1], disp_y=target_scan[6], disp_z=target_scan[7]
            )

        for lo in self.exp.linked_obstacles:
            data = lo.filter_linked_obstacle(
                scan_axis_index=scan_axis_index, diff_vectors=data[0],
                initial_axes_angles=initial_angles, diff_angles=data[3], mode='shade',
                directions_axes=self.exp.axes_directions, rotation_axes=self.exp.axes_rotations, data=data
            )

        if hasattr(self.exp, 'obstacles') and self.exp.obstacles:
            for obstacle in self.exp.obstacles:
                obstacle_ = self.exp._create_obstacle(obstacle)
                data = obstacle_.filter(diff_vecs=data[0], data=data, mode='shade')


        sim_hkl_orig = data[2].astype(int)


        if data[3].shape[1] > 1:
            sim_scan_angles = data[3][:, scan_axis_index]
        else:
            sim_scan_angles = data[3].flatten()

        sim_scan_angles = np.mod(np.rad2deg(sim_scan_angles), 360.0)

        sim_encoded = encode_hkl(sim_hkl_orig).flatten()


        sim_indices = [self.encoded_to_dense_map.get(code) for code in sim_encoded]

        valid_mask = [i is not None for i in sim_indices]

        self.sim_hkl_indices = np.array([i for i in sim_indices if i is not None], dtype=np.int32)
        self.sim_angles = sim_scan_angles[valid_mask]


        sort_order = np.argsort(self.sim_angles)
        self.sim_angles = self.sim_angles[sort_order]
        self.sim_hkl_indices = self.sim_hkl_indices[sort_order]

    def calculate_map(self, step_x=5.0, step_y=5.0, metric='completeness'):
        cache_key = (self.target_scan_index, metric, step_x, step_y)
        if cache_key in self._maps_data_cache:
            return self._maps_data_cache[cache_key]

        if self.target_scan_index is None:
            raise ValueError("Скан не выбран.")

        target_scan = self.exp.scans[self.target_scan_index]



        collision_map = self.exp.collision_handler.generate_sweep_map(
            exp_inst=self.exp, scan=target_scan, step=step_x, default_max=360.0
        )

        map_angles = np.array([item[0] for item in collision_map])
        collision_limits = np.array([item[1] for item in collision_map])

        X_out, Y_out, Z_out = [], [], []

        total_possible = self.total_unique_refs_count
        nonzero_base = np.count_nonzero(self.base_counts)
        sum_base_measurements = np.sum(self.base_counts)


        for i, start_angle in enumerate(map_angles):
            max_sweep = collision_limits[i]


            if max_sweep <= 0.001:

                X_out.append(start_angle)
                Y_out.append(0)
                Z_out.append(np.nan)
                continue

            y_sweeps = np.arange(step_y, max_sweep + 0.001, step_y)


            norm_start_angle = start_angle % 360.0

            for sweep in y_sweeps:

                norm_end_angle = (norm_start_angle + sweep) % 360.0


                if sweep >= 360.0:
                    current_scan_indices = self.sim_hkl_indices
                else:
                    if norm_start_angle < norm_end_angle:

                        idx_start = np.searchsorted(self.sim_angles, norm_start_angle, side='left')
                        idx_end = np.searchsorted(self.sim_angles, norm_end_angle, side='right')
                        current_scan_indices = self.sim_hkl_indices[idx_start:idx_end]
                    else:


                        idx_start_1 = np.searchsorted(self.sim_angles, norm_start_angle, side='left')
                        indices_1 = self.sim_hkl_indices[idx_start_1:]


                        idx_end_2 = np.searchsorted(self.sim_angles, norm_end_angle, side='right')
                        indices_2 = self.sim_hkl_indices[:idx_end_2]

                        current_scan_indices = np.concatenate((indices_1, indices_2))


                unique_inds_in_window = np.unique(current_scan_indices)

                if metric == 'completeness':
                    base_vals = self.base_counts[unique_inds_in_window]
                    newly_covered = np.count_nonzero(base_vals == 0)
                    val = (nonzero_base + newly_covered) / total_possible * 100.0
                elif metric == 'redundancy':
                    total_measurements = sum_base_measurements + len(current_scan_indices)
                    base_vals = self.base_counts[unique_inds_in_window]
                    newly_covered = np.count_nonzero(base_vals == 0)
                    total_unique_measured = nonzero_base + newly_covered
                    val = 0.0 if total_unique_measured == 0 else total_measurements / total_unique_measured
                elif metric == 'multiplicity':
                    total_measurements = sum_base_measurements + len(current_scan_indices)
                    val = total_measurements / total_possible
                else:
                    val = np.nan


                X_out.append(start_angle)
                Y_out.append(sweep)
                Z_out.append(val)

        result = (np.array(X_out), np.array(Y_out), np.array(Z_out))
        self._maps_data_cache[cache_key] = result
        return result

class OptimizationPlotter:
    def __init__(self, optimizer):
        self.opt = optimizer
        self.opt.set_plotter(self)
        self._figures_cache = {}

    def clear_cache(self):
        self._figures_cache.clear()

    def get_heatmap_figure(self, metric='completeness', step_x=5.0, step_y=5.0):
        """
        Генерирует 2D тепловую карту с отметкой текущего положения скана.
        """
        scan_idx = self.opt.target_scan_index
        if scan_idx is None:
            return go.Figure()

        cache_key = (scan_idx, metric, step_x, step_y, 'heatmap')

        if cache_key in self._figures_cache:
            return self._figures_cache[cache_key]


        X_flat, Y_flat, Z_flat = self.opt.calculate_map(step_x=step_x, step_y=step_y, metric=metric)

        x_coords = np.unique(X_flat)
        y_coords = np.unique(Y_flat)
        z_matrix = np.full((len(y_coords), len(x_coords)), np.nan)

        for x, y, z in zip(X_flat, Y_flat, Z_flat):

            ix = np.where(x_coords == x)[0][0]
            iy = np.where(y_coords == y)[0][0]
            z_matrix[iy, ix] = z

        titles = {'completeness': 'Completeness (%)', 'redundancy': 'Redundancy', 'multiplicity': 'Multiplicity'}
        z_title = titles.get(metric, metric)
        scan_name = self.opt.exp.axes_names[self.opt.exp.scans[scan_idx][4]]


        current_scan = self.opt.exp.scans[scan_idx]
        axis_idx = current_scan[4]

        current_start_angle = current_scan[3][axis_idx]

        current_sweep = current_scan[5]


        fig = go.Figure()


        fig.add_trace(go.Heatmap(
            z=z_matrix, x=x_coords, y=y_coords,
            colorscale='Viridis',
            colorbar=dict(title=z_title),
            hovertemplate="Start: %{x:.1f}°<br>Sweep: %{y:.1f}°<br>Value: %{z:.2f}<extra></extra>",
            connectgaps=False
        ))



        fig.add_trace(go.Scatter(
            x=[current_start_angle],
            y=[current_sweep],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                symbol='cross',
                line=dict(color='white', width=2)
            ),
            name='Current Params',
            hoverinfo='skip'
        ))


        fig.update_layout(
            shapes=[

                dict(
                    type="line",
                    x0=current_start_angle, y0=0,
                    x1=current_start_angle, y1=current_sweep,
                    line=dict(color="red", width=2, dash="dash"),
                ),

                dict(
                    type="line",
                    x0=0, y0=current_sweep,
                    x1=current_start_angle, y1=current_sweep,
                    line=dict(color="red", width=2, dash="dash"),
                ),
            ],
            title=f"Optimization: Scan {scan_idx} ({scan_name})<br>"
                  f"Current: Start {current_start_angle:.1f}°, Sweep {current_sweep:.1f}°",
            xaxis_title="Start Angle (°)",
            yaxis_title="Sweep Length (°)",
            template="plotly_white",

            xaxis=dict(range=[0, 360]),

        )

        self._figures_cache[cache_key] = fig
        return fig

    def get_3d_map_figure(self, metric='completeness', step_x=5.0, step_y=5.0):
        """
        Генерирует или возвращает из кэша 3D график для ТЕКУЩЕГО выбранного в оптимизаторе скана.
        """
        scan_idx = self.opt.target_scan_index
        if scan_idx is None:
            return go.Figure()

        cache_key = (scan_idx, metric, step_x, step_y)

        if cache_key in self._figures_cache:
            return self._figures_cache[cache_key]

        X, Y, Z = self.opt.calculate_map(step_x=step_x, step_y=step_y, metric=metric)

        scan_name = self.opt.exp.axes_names[self.opt.exp.scans[scan_idx][4]]

        titles = {
            'completeness': 'Completeness (%)',
            'redundancy': 'Redundancy',
            'multiplicity': 'Multiplicity'
        }
        z_title = titles.get(metric, metric)

        trace = go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=dict(
                size=4,
                color=Z,
                colorscale='Viridis',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title=z_title)
            ),
            hovertemplate=(
                    "Start Angle: %{x:.1f}°<br>" +
                    "Sweep: %{y:.1f}°<br>" +
                    f"{z_title}: %{{z:.2f}}<extra></extra>"
            )
        )

        layout = go.Layout(
            title=f"Optimization Analysis: Scan {scan_idx} ({scan_name})",
            scene=dict(
                xaxis_title='Start Angle (°)',
                yaxis_title='Sweep Length (°)',
                zaxis_title=z_title,

            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig = go.Figure(data=[trace], layout=layout)

        self._figures_cache[cache_key] = fig

        return fig


class OptimizationManager:
    def __init__(self, experiment):
        self.experiment = experiment

        self._optimizer = ScanOptimizer(experiment)
        self._plotter = OptimizationPlotter(self._optimizer)


        self._current_metric = 'completeness'
        self._current_view_mode = '2d'


        self._step_x = 5.0
        self._step_y = 5.0

    def select_scan(self, scan_index: int):
        """
        1. Выбирает скан в оптимизаторе.
        2. Возвращает актуальный график.
        """
        self._optimizer.select_scan(scan_index)
        return self._get_current_figure()

    def apply_parameters(self, start_angle: float, sweep: float):
        """
        1. Применяет новые параметры (Preview).
        2. Возвращает обновленный график с перемещенным маркером.
        """
        self._optimizer.apply_parameters(start_angle, sweep)
        return self._get_current_figure()

    def revert_changes(self):
        """
        1. Откатывает изменения к исходному состоянию скана.
        2. Возвращает график с исходным положением маркера.
        """
        self._optimizer.revert_changes()
        return self._get_current_figure()

    def toggle_base_run(self, scan_index: int, active: bool):
        """
        Включает или исключает скан из расчета фона и возвращает обновленный график.
        """
        if active:
            self._optimizer.include_base_run(scan_index)
        else:
            self._optimizer.exclude_base_run(scan_index)

        return self._get_current_figure()

    def select_graph_type(self, metric: str):
        """
        1. Меняет отображаемую метрику (completeness, redundancy, multiplicity).
        2. Возвращает перестроенный график.
        """
        valid_metrics = ['completeness', 'redundancy', 'multiplicity']
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of {valid_metrics}")

        self._current_metric = metric
        return self._get_current_figure()

    def _get_current_figure(self):
        """Вспомогательный метод для генерации графика на основе текущего состояния."""

        if self._current_view_mode == '2d':
            return self._plotter.get_heatmap_figure(
                metric=self._current_metric,
                step_x=self._step_x,
                step_y=self._step_y
            )
        else:
            return self._plotter.get_3d_map_figure(
                metric=self._current_metric,
                step_x=self._step_x,
                step_y=self._step_y
            )
