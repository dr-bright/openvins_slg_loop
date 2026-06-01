import multiprocessing as mp
import os
import queue
import sys

from fire import Fire
import cv2 as cv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

try:
    import rosbag
    from cv_bridge import CvBridge
except ImportError:
    rosbag = None
    CvBridge = None

def get_verbose_progress(verbose=None, progress=None):
    if verbose is True:
        def verbose(*args, **kwargs):
            kwargs['file'] = sys.stderr
            return print(*args, **kwargs)
    if progress is True:
        progress = verbose
    return verbose, progress


def read_txt(path):
    with open(path) as file:
        file.read(1)
        data = pd.read_csv(file, sep=' ', skipinitialspace=True)
    return data.rename(columns={data.columns[0]: 'ts'})


def trajs_slice(trajs, start=None, stop=None):
    ts_begin = min(map(lambda traj: traj.ts.iloc[0], trajs))
    ts_end = max(map(lambda traj: traj.ts.iloc[-1], trajs))
    if start is not None or stop is not None:
        for i, traj in enumerate(trajs):
            idx = ~np.isnan(traj.ts)
            if start is not None:
                if start < 0:
                    start = ts_end + start
                idx[traj.ts < start] = False
            if stop is not None:
                if stop < 0:
                    stop = ts_end + stop
                idx[traj.ts >= stop] = False
            trajs[i] = traj.loc[idx]
    return ts_begin, ts_end


def make_txyz(fig, add=None):
    front_view = fig.add_subplot(2, 2, 1)
    front_view.set_xlabel('x, m')
    front_view.set_ylabel('z, m')
    front_view.set_title('Главный вид (спереди)')
    front_view.set_aspect('equal')

    left_view = fig.add_subplot(2, 2, 2)
    left_view.set_xlabel('y, m')
    left_view.set_ylabel('z, m')
    left_view.invert_xaxis()
    left_view.set_title('Вид слева')
    left_view.set_aspect('equal')

    top_view = fig.add_subplot(2, 2, 3)
    top_view.set_xlabel('x, m')
    top_view.set_ylabel('y, m')
    top_view.set_title('Вид сверху')
    top_view.set_aspect('equal')

    if add == 'vid':
        video_view = fig.add_subplot(2, 2, 4)
        video_view.axis('off')
        video_view.set_title('Видео')
    elif add == '3d':
        video_view = fig.add_subplot(2, 2, 4, projection='3d')
        video_view.set_xlabel('x, m')
        video_view.set_ylabel('y, m')
        video_view.set_zlabel('z, m')
        video_view.set_title('3D projection')
    else:
        video_view = None
    return front_view, left_view, top_view, video_view


def make_vertical(fig, add=None):
    if add == 'vid':
        top_view = fig.add_subplot(1, 2, 1)
    else:
        top_view = fig.add_subplot(1, 1, 1)
    top_view.set_xlabel('x, m')
    top_view.set_ylabel('y, m')
    top_view.set_title('Вертикальная проекция')
    top_view.set_aspect('equal')

    video_view = None
    if add == 'vid':
        video_view = fig.add_subplot(1, 2, 2)
        video_view.axis('off')
        video_view.set_title('Видео')
    return top_view, video_view


class AnimationRenderer:
    def __init__(self, *trajectories, traj_names=None, bag=None, topic=None, save=None,
                 fps=None, size=(20, 15), dpi=100, forward_axis='z', vertical=False):
        if not trajectories:
            raise ValueError('At least one trajectory is required')

        self.traj_arrays = [traj.to_numpy() for traj in trajectories]
        self.traj_names = tuple(traj_names or [f'traj {i}' for i in range(len(trajectories))])
        self.bag = bag
        self.topic = topic
        self.save = save
        self.size = size
        self.dpi = dpi
        self.forward_axis = self._parse_forward_axis(forward_axis)
        self.vertical = vertical

        self.ts_begin = min(traj.ts.iloc[0] for traj in trajectories)
        self.ts_end = max(traj.ts.iloc[-1] for traj in trajectories)
        if fps is None:
            dt = np.concatenate([np.diff(traj.ts) for traj in trajectories]).mean()
            fps = round(1 / dt)
        self.fps = fps
        self.total = int(self.fps * (self.ts_end - self.ts_begin))
        if self.total <= 0:
            raise RuntimeError('No animation frames to render')

        self.frame_ts = np.linspace(self.ts_begin, self.ts_end, self.total, endpoint=False)
        self.points = self._precompute_points(trajectories)
        self.directions = self._precompute_directions(trajectories)
        self.arrow_scale = self._estimate_arrow_scale()

        self._fig = None
        self._canvas = None
        self._scatxy = None
        self._scatyz = None
        self._scatxz = None
        self._arrowxy = None
        self._arrowyz = None
        self._arrowxz = None
        self._view3d = None
        self._scat3d = None
        self._arrow3d = None
        self._video_artist = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in (
            '_fig',
            '_canvas',
            '_scatxy',
            '_scatyz',
            '_scatxz',
            '_arrowxy',
            '_arrowyz',
            '_arrowxz',
            '_view3d',
            '_scat3d',
            '_arrow3d',
            '_video_artist',
        ):
            state[key] = None
        return state

    @property
    def frame_size(self):
        width = int(self.size[0] * self.dpi)
        height = int(self.size[1] * self.dpi)
        return width, height

    def _precompute_points(self, trajectories):
        points = np.zeros((self.total, len(trajectories), 3), dtype=float)
        for i, traj in enumerate(trajectories):
            f = interp1d(
                traj.ts,
                traj.to_numpy(),
                axis=0,
                copy=False,
                assume_sorted=True,
                bounds_error=False,
            )
            points[:, i, :] = f(self.frame_ts)[:, 1:4]
        return points

    def _orientation_columns(self, traj):
        columns = {str(col).lower(): col for col in traj.columns}
        if all(key in columns for key in ('qx', 'qy', 'qz', 'qw')):
            return [columns[key] for key in ('qx', 'qy', 'qz', 'qw')]
        if traj.shape[1] >= 8:
            return list(traj.columns[4:8])
        return None

    def _parse_forward_axis(self, axis):
        if isinstance(axis, (tuple, list, np.ndarray)):
            vec = np.asarray(axis, dtype=float)
            if vec.shape != (3,):
                raise ValueError('forward_axis vector must have three elements')
        else:
            text = str(axis).strip().lower()
            sign = -1.0 if text.startswith('-') else 1.0
            name = text[1:] if text.startswith(('-', '+')) else text
            axes = {
                'x': np.array([1.0, 0.0, 0.0]),
                'y': np.array([0.0, 1.0, 0.0]),
                'z': np.array([0.0, 0.0, 1.0]),
            }
            if name not in axes:
                raise ValueError("forward_axis must be one of 'x', 'y', 'z', '-x', '-y', '-z' or a 3-vector")
            vec = sign * axes[name]
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            raise ValueError('forward_axis vector must be non-zero')
        return vec / norm

    def _precompute_directions(self, trajectories):
        directions = np.full((self.total, len(trajectories), 3), np.nan, dtype=float)
        found = False
        local_axis = self.forward_axis
        for i, traj in enumerate(trajectories):
            qcols = self._orientation_columns(traj)
            if qcols is None:
                continue
            f = interp1d(
                traj.ts,
                traj[qcols].to_numpy(),
                axis=0,
                copy=False,
                assume_sorted=True,
                bounds_error=False,
            )
            quat = f(self.frame_ts)
            norms = np.linalg.norm(quat, axis=1)
            valid = norms > 1e-12
            if not np.any(valid):
                continue
            quat[valid] = quat[valid] / norms[valid, None]
            qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

            # Treat the common TUM/OpenVINS output qx qy qz qw as local-to-world
            # orientation and project the selected local forward axis into world axes.
            r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
            r01 = 2.0 * (qx * qy - qz * qw)
            r02 = 2.0 * (qx * qz + qy * qw)
            r10 = 2.0 * (qx * qy + qz * qw)
            r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
            r12 = 2.0 * (qy * qz - qx * qw)
            r20 = 2.0 * (qx * qz - qy * qw)
            r21 = 2.0 * (qy * qz + qx * qw)
            r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
            directions[:, i, 0] = r00 * local_axis[0] + r01 * local_axis[1] + r02 * local_axis[2]
            directions[:, i, 1] = r10 * local_axis[0] + r11 * local_axis[1] + r12 * local_axis[2]
            directions[:, i, 2] = r20 * local_axis[0] + r21 * local_axis[1] + r22 * local_axis[2]
            directions[~valid, i, :] = np.nan
            found = True
        return directions if found else None

    def _estimate_arrow_scale(self):
        mins = np.nanmin(self.points.reshape(-1, 3), axis=0)
        maxs = np.nanmax(self.points.reshape(-1, 3), axis=0)
        span = np.nanmax(maxs - mins)
        if not np.isfinite(span) or span <= 0:
            return 1.0
        return 0.06 * span

    def _ensure_figure(self, video_frame=None):
        if self._fig is not None:
            return

        import matplotlib

        matplotlib.use('Agg', force=True)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=self.size, dpi=self.dpi)
        canvas = FigureCanvasAgg(fig)
        if self.vertical:
            top_view, video_view = make_vertical(fig, add='vid' if self.bag else None)
            front_view = None
            left_view = None
        else:
            front_view, left_view, top_view, video_view = make_txyz(fig, add='vid' if self.bag else '3d')

        for name, traj in zip(self.traj_names, self.traj_arrays):
            if top_view is not None:
                top_view.plot(traj[:, 1], traj[:, 2], label=name)
            if left_view is not None:
                left_view.plot(traj[:, 2], traj[:, 3], label=name)
            if front_view is not None:
                front_view.plot(traj[:, 1], traj[:, 3], label=name)
            if self.bag is None:
                if video_view is not None:
                    video_view.plot(traj[:, 1], traj[:, 2], traj[:, 3], label=name)
        if left_view is not None:
            left_view.legend()
        else:
            top_view.legend()

        if top_view is not None:
            self._scatxy = top_view.scatter(0, 0, c='r', zorder=10)
        if left_view is not None:
            self._scatyz = left_view.scatter(0, 0, c='r', zorder=10)
        if front_view is not None:
            self._scatxz = front_view.scatter(0, 0, c='r', zorder=10)
        if self.directions is not None:
            zeros = np.zeros(len(self.traj_arrays))
            if top_view is not None:
                self._arrowxy = top_view.quiver(zeros, zeros, zeros, zeros, color='r', angles='xy', scale_units='xy', scale=1, zorder=11)
            if left_view is not None:
                self._arrowyz = left_view.quiver(zeros, zeros, zeros, zeros, color='r', angles='xy', scale_units='xy', scale=1, zorder=11)
            if front_view is not None:
                self._arrowxz = front_view.quiver(zeros, zeros, zeros, zeros, color='r', angles='xy', scale_units='xy', scale=1, zorder=11)

        if video_view is not None:
            if self.bag:
                if video_frame is None:
                    video_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self._video_artist = video_view.imshow(cv.cvtColor(video_frame, cv.COLOR_BGR2RGB))
            else:
                self._view3d = video_view
                self._scat3d = video_view.scatter([], [], [], c='r', zorder=10)

        self._fig = fig
        self._canvas = canvas

    def _iter_video_frames(self):
        if self.bag is None:
            for _ in range(self.total):
                yield None
            return
        yield from self._iter_bag_frames()

    def _infer_bag_image_topic(self, bag):
        for topic, msg, _ in bag.read_messages():
            if getattr(msg, '_type', None) == 'sensor_msgs/Image':
                return topic
        raise RuntimeError(f'Could not infer image topic from bag: {self.bag}')

    def _decode_bag_image(self, bridge, msg):
        if msg.encoding == 'bgr8':
            return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if msg.encoding == 'rgb8':
            rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        if msg.encoding in ('mono8', '8UC1'):
            gray = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        if msg.encoding in ('mono16', '16UC1'):
            mono16 = bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
            gray = cv.convertScaleAbs(mono16, alpha=1.0 / 256.0)
            return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _iter_bag_frames(self):
        if rosbag is None or CvBridge is None:
            raise RuntimeError('Bag video source requires rosbag and cv_bridge Python modules')

        bag = rosbag.Bag(self.bag, 'r')
        bridge = CvBridge()
        try:
            topic = self.topic or self._infer_bag_image_topic(bag)
            messages = bag.read_messages(topics=[topic])
            current = None
            current_ts = None

            try:
                topic_name, msg, _ = next(messages)
                current = self._decode_bag_image(bridge, msg)
                current_ts = msg.header.stamp.to_sec()
            except StopIteration:
                raise RuntimeError(f'No images on topic {topic} in bag {self.bag}')

            for ts in self.frame_ts:
                while current_ts is not None and current_ts < ts:
                    try:
                        topic_name, msg, _ = next(messages)
                        current = self._decode_bag_image(bridge, msg)
                        current_ts = msg.header.stamp.to_sec()
                    except StopIteration:
                        break
                yield None if current is None else current.copy()
        finally:
            bag.close()

    def _task(self, frame_id, video_frames):
        return frame_id, next(video_frames)

    def render_frame(self, frame_id, video_frame=None):
        self._ensure_figure(video_frame)

        pts = self.points[frame_id]
        if self._scatxy is not None:
            self._scatxy.set_offsets(pts[:, [0, 1]])
        if self._scatyz is not None:
            self._scatyz.set_offsets(pts[:, [1, 2]])
        if self._scatxz is not None:
            self._scatxz.set_offsets(pts[:, [0, 2]])
        if self._scat3d is not None:
            self._scat3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        if self.directions is not None:
            dirs = self.directions[frame_id] * self.arrow_scale
            valid = np.all(np.isfinite(dirs), axis=1)
            arrow_pts = pts[valid]
            arrow_dirs = dirs[valid]
            if self._arrowxy is not None:
                self._arrowxy.set_offsets(arrow_pts[:, [0, 1]])
                self._arrowxy.set_UVC(arrow_dirs[:, 0], arrow_dirs[:, 1])
            if self._arrowyz is not None:
                self._arrowyz.set_offsets(arrow_pts[:, [1, 2]])
                self._arrowyz.set_UVC(arrow_dirs[:, 1], arrow_dirs[:, 2])
            if self._arrowxz is not None:
                self._arrowxz.set_offsets(arrow_pts[:, [0, 2]])
                self._arrowxz.set_UVC(arrow_dirs[:, 0], arrow_dirs[:, 2])

            if self._view3d is not None:
                if self._arrow3d is not None:
                    self._arrow3d.remove()
                    self._arrow3d = None
                if len(arrow_pts):
                    self._arrow3d = self._view3d.quiver(
                        arrow_pts[:, 0], arrow_pts[:, 1], arrow_pts[:, 2],
                        arrow_dirs[:, 0], arrow_dirs[:, 1], arrow_dirs[:, 2],
                        color='r',
                        length=1.0,
                        normalize=False,
                    )

        if self._video_artist is not None and video_frame is not None:
            self._video_artist.set_data(cv.cvtColor(video_frame, cv.COLOR_BGR2RGB))

        self._canvas.draw()
        width, height = self._canvas.get_width_height()
        rgb = np.frombuffer(self._canvas.tostring_rgb(), dtype=np.uint8)
        rgb = rgb.reshape((height, width, 3))
        return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

    def _open_writer(self):
        if not self.save:
            raise ValueError('save path is required for VideoWriter output')

        width, height = self.frame_size
        ext = os.path.splitext(self.save)[1].lower()
        codec_options = {
            '.avi': ['XVID', 'MJPG'],
            '.mp4': ['avc1', 'H264', 'X264', 'mp4v', 'MJPG'],
            '.m4v': ['avc1', 'H264', 'X264', 'mp4v', 'MJPG'],
            '.mov': ['avc1', 'H264', 'X264', 'mp4v', 'MJPG'],
        }
        for codec in codec_options.get(ext, ['avc1', 'H264', 'X264', 'mp4v', 'MJPG']):
            writer = cv.VideoWriter(
                self.save,
                cv.VideoWriter_fourcc(*codec),
                self.fps,
                (width, height),
            )
            if writer.isOpened():
                return writer
            writer.release()
        raise RuntimeError(f'Could not open VideoWriter for {self.save}')

    def __call__(self, jobs=None, progress=True):
        if jobs is None:
            jobs = os.cpu_count() or 1
        jobs = max(1, min(jobs, self.total))

        if not self.save:
            self.preview()
            return

        writer = self._open_writer()
        ctx = mp.get_context()
        frame_queue = ctx.Queue(maxsize=jobs * 3)
        output_queue = ctx.Queue(maxsize=jobs * 3)
        workers = [
            ctx.Process(target=_render_worker, args=(self, frame_queue, output_queue))
            for _ in range(jobs)
        ]

        for worker in workers:
            worker.start()

        next_input = 0
        next_output = 0
        pending = {}
        errors = []
        video_frames = self._iter_video_frames()

        try:
            for _ in range(min(jobs * 3, self.total)):
                frame_queue.put(self._task(next_input, video_frames))
                next_input += 1

            with tqdm(total=self.total, disable=not progress) as pbar:
                while next_output < self.total:
                    try:
                        frame_id, frame, error = output_queue.get(timeout=0.1)
                    except queue.Empty:
                        if any(not worker.is_alive() and worker.exitcode not in (0, None) for worker in workers):
                            raise RuntimeError('A render worker exited unexpectedly')
                        continue

                    if error is not None:
                        errors.append(error)
                        break

                    pending[frame_id] = frame
                    while next_output in pending:
                        writer.write(pending.pop(next_output))
                        next_output += 1
                        pbar.update()
                        if next_input < self.total:
                            frame_queue.put(self._task(next_input, video_frames))
                            next_input += 1

            if errors:
                raise RuntimeError(errors[0])
        finally:
            for _ in workers:
                try:
                    frame_queue.put_nowait(None)
                except queue.Full:
                    break
            for worker in workers:
                worker.join(timeout=1)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            writer.release()

    def preview(self):
        video_frames = self._iter_video_frames()
        for frame_id in tqdm(range(self.total)):
            frame = self.render_frame(frame_id, next(video_frames))
            cv.imshow('Animation', frame)
            if cv.waitKey(round(1000 / self.fps)) in (ord('q'), 27):
                break
        cv.destroyAllWindows()


def _render_worker(renderer, frame_queue, output_queue):
    while True:
        task = frame_queue.get()
        if task is None:
            break
        frame_id, video_frame = task
        try:
            frame = renderer.render_frame(frame_id, video_frame)
        except Exception as exc:
            output_queue.put((frame_id, None, repr(exc)))
            break
        output_queue.put((frame_id, frame, None))

def animate_cli(traj_name, *traj_names, bag: str = None,
                topic: str = None, save: str = None, start=None, stop=None,
                fps=None, jobs=4, forward_axis='-z', vertical=False,
                verbose=True):
    verbose = get_verbose_progress(verbose)[0]
    verbose and verbose(f'Reading files {traj_names}')
    trajs = [read_txt(traj) for traj in (traj_name, *traj_names)]
    trajs_slice(trajs, start, stop)

    renderer = AnimationRenderer(
        *trajs,
        traj_names=traj_names,
        bag=bag,
        topic=topic,
        save=save,
        fps=fps,
        forward_axis=forward_axis,
        vertical=vertical,
    )
    verbose and verbose(f'Rendering {renderer.total} frames')
    renderer(progress=bool(verbose), jobs=jobs)


Fire(animate_cli)
