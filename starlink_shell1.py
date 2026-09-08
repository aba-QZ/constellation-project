#!/usr/bin/env python3
"""Starlink Shell 1 Walker-Delta constellation model.

This is a NumPy/Matplotlib translation of the supplied MATLAB script.  The
model is intentionally idealized: circular two-body orbits, optional first-
order J2 secular RAAN drift, a spherical Earth, and GMST = 0 at t = 0.

It reproduces the commonly used public reconstruction 1584 / 72 / 39.  For
positions at a real UTC epoch, use TLE data with an SGP4 propagator instead.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


# Earth constants (kilometres and seconds)
MU_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137
EARTH_ROTATION_RAD_S = 7.2921159e-5
J2 = 1.08262668e-3

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ConstellationConfig:
    """Parameters for a circular Walker-Delta constellation."""

    height_km: float = 550.0
    inclination_deg: float = 53.0
    satellite_count: int = 1584
    plane_count: int = 72
    phase_factor: int = 39
    use_j2: bool = True

    def validate(self) -> None:
        if self.satellite_count % self.plane_count != 0:
            raise ValueError("卫星总数必须能被轨道面数整除。")
        if not 0 <= self.phase_factor < self.plane_count:
            raise ValueError("Walker 相位因子 F 应满足 0 <= F < P。")
        if self.height_km <= 0:
            raise ValueError("轨道高度必须大于 0。")
        if not 0 <= self.inclination_deg <= 180:
            raise ValueError("轨道倾角必须位于 [0, 180] deg。")

    @property
    def satellites_per_plane(self) -> int:
        return self.satellite_count // self.plane_count

    @property
    def semi_major_axis_km(self) -> float:
        return EARTH_RADIUS_KM + self.height_km

    @property
    def inclination_rad(self) -> float:
        return np.deg2rad(self.inclination_deg).item()

    @property
    def mean_motion_rad_s(self) -> float:
        return np.sqrt(MU_KM3_S2 / self.semi_major_axis_km**3).item()

    @property
    def orbital_period_s(self) -> float:
        return 2.0 * np.pi / self.mean_motion_rad_s


@dataclass(frozen=True)
class ConstellationState:
    """Initial geometry and propagated positions at one simulation time."""

    time_s: float
    raan0_rad: FloatArray
    argument_of_latitude0_rad: FloatArray
    raan_rad: FloatArray
    argument_of_latitude_rad: FloatArray
    x_eci_km: FloatArray
    y_eci_km: FloatArray
    z_eci_km: FloatArray
    x_ecef_km: FloatArray
    y_ecef_km: FloatArray
    z_ecef_km: FloatArray
    latitude_deg: FloatArray
    longitude_deg: FloatArray

    @staticmethod
    def matlab_flatten(values: FloatArray) -> FloatArray:
        """Match MATLAB's column-major ``A(:)`` ordering."""

        return values.ravel(order="F")


def initial_walker_geometry(
    config: ConstellationConfig,
) -> tuple[FloatArray, FloatArray]:
    """Return initial RAAN and argument-of-latitude matrices.

    Walker-Delta convention:

        u(i, j) = 2*pi * (j/S + i*F/T)

    where i = 0..P-1 and j = 0..S-1.
    """

    plane_ids = np.arange(config.plane_count, dtype=float)[:, None]
    slot_ids = np.arange(config.satellites_per_plane, dtype=float)[None, :]

    raan0 = 2.0 * np.pi * plane_ids / config.plane_count
    u0 = 2.0 * np.pi * (
        slot_ids / config.satellites_per_plane
        + plane_ids * config.phase_factor / config.satellite_count
    )
    return raan0, np.mod(u0, 2.0 * np.pi)


def raan_drift_rate_rad_s(config: ConstellationConfig) -> float:
    """First-order secular J2 RAAN drift for a circular orbit."""

    if not config.use_j2:
        return 0.0
    return (
        -1.5
        * config.mean_motion_rad_s
        * J2
        * (EARTH_RADIUS_KM / config.semi_major_axis_km) ** 2
        * np.cos(config.inclination_rad)
    ).item()


def eci_positions(
    radius_km: float,
    inclination_rad: float,
    raan_rad: FloatArray,
    argument_of_latitude_rad: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert circular-orbit elements to ECI Cartesian coordinates."""

    cos_raan = np.cos(raan_rad)
    sin_raan = np.sin(raan_rad)
    cos_u = np.cos(argument_of_latitude_rad)
    sin_u = np.sin(argument_of_latitude_rad)
    cos_inc = np.cos(inclination_rad)
    sin_inc = np.sin(inclination_rad)

    x = radius_km * (cos_raan * cos_u - sin_raan * sin_u * cos_inc)
    y = radius_km * (sin_raan * cos_u + cos_raan * sin_u * cos_inc)
    z = radius_km * (sin_u * sin_inc)
    return x, y, z


def propagate(config: ConstellationConfig, time_s: float) -> ConstellationState:
    """Propagate the idealized constellation to ``time_s``."""

    config.validate()
    raan0, u0 = initial_walker_geometry(config)
    omega_dot = raan_drift_rate_rad_s(config)

    raan = np.mod(raan0 + omega_dot * time_s, 2.0 * np.pi)
    u = np.mod(u0 + config.mean_motion_rad_s * time_s, 2.0 * np.pi)

    x, y, z = eci_positions(
        config.semi_major_axis_km,
        config.inclination_rad,
        raan,
        u,
    )

    # Simplified ECI -> ECEF rotation, with GMST = 0 at t = 0.
    theta = np.mod(EARTH_ROTATION_RAD_S * time_s, 2.0 * np.pi)
    x_ecef = np.cos(theta) * x + np.sin(theta) * y
    y_ecef = -np.sin(theta) * x + np.cos(theta) * y
    z_ecef = z.copy()

    radius_ecef = np.sqrt(x_ecef**2 + y_ecef**2 + z_ecef**2)
    latitude = np.rad2deg(np.arcsin(z_ecef / radius_ecef))
    longitude = np.rad2deg(np.arctan2(y_ecef, x_ecef))

    if x.size != config.satellite_count:
        raise AssertionError("生成的卫星数量和设定值不一致。")

    return ConstellationState(
        time_s=time_s,
        raan0_rad=raan0,
        argument_of_latitude0_rad=u0,
        raan_rad=raan,
        argument_of_latitude_rad=u,
        x_eci_km=x,
        y_eci_km=y,
        z_eci_km=z,
        x_ecef_km=x_ecef,
        y_ecef_km=y_ecef,
        z_ecef_km=z_ecef,
        latitude_deg=latitude,
        longitude_deg=longitude,
    )


def print_report(config: ConstellationConfig, state: ConstellationState) -> None:
    """Print the same key geometry and numerical checks as the MATLAB code."""

    satellites_per_plane = config.satellites_per_plane
    delta_raan_deg = 360.0 / config.plane_count
    delta_sat_deg = 360.0 / satellites_per_plane
    delta_phase_deg = 360.0 * config.phase_factor / config.satellite_count
    omega_dot = raan_drift_rate_rad_s(config)
    latitude = state.matlab_flatten(state.latitude_deg)
    longitude = state.matlab_flatten(state.longitude_deg)

    print()
    print("=" * 60)
    print("      Starlink Shell 1 Walker-Delta 星座模型")
    print("=" * 60)
    print(f"轨道高度 h              = {config.height_km:.3f} km")
    print(f"半长轴 a                = {config.semi_major_axis_km:.3f} km")
    print(f"轨道倾角 i              = {config.inclination_deg:.3f} deg")
    print()
    print(f"轨道面数 P              = {config.plane_count:d}")
    print(f"每面卫星数 S            = {satellites_per_plane:d}")
    print(f"卫星总数 T              = {config.satellite_count:d}")
    print(f"Walker 相位因子 F       = {config.phase_factor:d}")
    print()
    print(
        "Walker 构型             = "
        f"{config.satellite_count:d} / {config.plane_count:d} / "
        f"{config.phase_factor:d}"
    )
    print(f"相邻轨道面 RAAN 间隔    = {delta_raan_deg:.6f} deg")
    print(f"同轨道面卫星角间隔      = {delta_sat_deg:.6f} deg")
    print(f"相邻轨道面相位偏移      = {delta_phase_deg:.6f} deg")
    print(f"相位偏移 / 面内间隔     = {delta_phase_deg / delta_sat_deg:.6f}")
    print()
    print(f"轨道周期                = {config.orbital_period_s / 60.0:.6f} min")
    print(f"平均运动                = {config.mean_motion_rad_s:.9e} rad/s")
    print()
    print(f"J2 RAAN 长期漂移        = {'开启' if config.use_j2 else '关闭'}")
    print(f"RAAN 漂移率             = {omega_dot:.9e} rad/s")
    print(f"RAAN 漂移率             = {np.rad2deg(omega_dot) * 86400.0:.6f} deg/day")
    print()
    print(f"仿真时刻 t              = {state.time_s:.3f} s")
    print(f"                        = {state.time_s / 60.0:.3f} min")
    print(f"                        = {state.time_s / 86400.0:.6f} day")
    print()
    print(f"实际生成卫星数          = {state.x_eci_km.size:d}")
    print(f"星下点纬度范围          = [{latitude.min():.6f}, {latitude.max():.6f}] deg")
    print(f"理论纬度极限            = +/- {config.inclination_deg:.3f} deg")
    print(f"星下点经度范围          = [{longitude.min():.6f}, {longitude.max():.6f}] deg")
    print("=" * 60)
    print()
    print("---------------- Walker 几何关系 ----------------")
    print("同一轨道面相邻卫星：")
    print(f"    Delta u_sat   = {delta_sat_deg:.6f} deg")
    print()
    print("相邻轨道面：")
    print(f"    Delta RAAN    = {delta_raan_deg:.6f} deg")
    print(f"    Delta u_phase = {delta_phase_deg:.6f} deg")
    print()
    print("换算成卫星槽位：")
    print(
        f"    Delta slot    = F/P = {config.phase_factor:d}/"
        f"{config.plane_count:d} = "
        f"{config.phase_factor / config.plane_count:.6f} slot"
    )
    print()
    print(
        "也就是说，相邻轨道面的卫星沿轨方向约错开 "
        f"{100.0 * config.phase_factor / config.plane_count:.2f}% 个槽位。"
    )
    print("-" * 49)


def configure_matplotlib() -> None:
    """Configure common Windows CJK fonts and correct minus-sign rendering."""

    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def earth_mesh(resolution: int = 60) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate a spherical-Earth surface mesh."""

    azimuth = np.linspace(0.0, 2.0 * np.pi, resolution + 1)
    polar = np.linspace(0.0, np.pi, resolution + 1)
    azimuth_grid, polar_grid = np.meshgrid(azimuth, polar)
    x = EARTH_RADIUS_KM * np.sin(polar_grid) * np.cos(azimuth_grid)
    y = EARTH_RADIUS_KM * np.sin(polar_grid) * np.sin(azimuth_grid)
    z = EARTH_RADIUS_KM * np.cos(polar_grid)
    return x, y, z


def style_3d_axes(ax: object, axis_limit: float) -> None:
    ax.set_xlabel(r"$X_{ECI}$ [km]")
    ax.set_ylabel(r"$Y_{ECI}$ [km]")
    ax.set_zlabel(r"$Z_{ECI}$ [km]")
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=20.0, azim=30.0)
    ax.grid(True)


def draw_earth(ax: object) -> None:
    x_sphere, y_sphere, z_sphere = earth_mesh()
    ax.plot_surface(
        x_sphere,
        y_sphere,
        z_sphere,
        color=(0.30, 0.50, 0.78),
        edgecolor="none",
        alpha=0.55,
        shade=True,
    )


def draw_orbit_planes(
    ax: object,
    config: ConstellationConfig,
    raan_rad: FloatArray,
    *,
    color: tuple[float, float, float] = (0.82, 0.82, 0.82),
    linewidth: float = 0.35,
) -> None:
    u_orbit = np.linspace(0.0, 2.0 * np.pi, 360)
    for omega in raan_rad[:, 0]:
        x, y, z = eci_positions(
            config.semi_major_axis_km,
            config.inclination_rad,
            np.asarray([[omega]]),
            u_orbit[None, :],
        )
        ax.plot(x.ravel(), y.ravel(), z.ravel(), color=color, linewidth=linewidth)


def create_static_plots(
    config: ConstellationConfig,
    state: ConstellationState,
) -> list[tuple[str, object]]:
    """Create the four figures from the MATLAB implementation."""

    import matplotlib.pyplot as plt

    configure_matplotlib()
    figures: list[tuple[str, object]] = []

    # Figure 1: 3D Earth, orbit planes, and satellites.
    fig1 = plt.figure(figsize=(9.0, 7.0), facecolor="white")
    ax1 = fig1.add_subplot(111, projection="3d")
    draw_earth(ax1)
    draw_orbit_planes(ax1, config, state.raan_rad)
    ax1.scatter(
        state.matlab_flatten(state.x_eci_km),
        state.matlab_flatten(state.y_eci_km),
        state.matlab_flatten(state.z_eci_km),
        s=14,
        c="red",
        depthshade=False,
    )
    style_3d_axes(ax1, 1.10 * config.semi_major_axis_km)
    ax1.set_title(
        "Starlink Shell 1  |  "
        f"{config.satellite_count}/{config.plane_count}/{config.phase_factor}"
        f"  |  t = {state.time_s / 60.0:.2f} min"
    )
    fig1.tight_layout()
    figures.append(("01_starlink_shell1_3d.png", fig1))

    # Figure 2: sub-satellite point distribution.
    fig2, ax2 = plt.subplots(figsize=(10.0, 4.5), facecolor="white")
    ax2.scatter(
        state.matlab_flatten(state.longitude_deg),
        state.matlab_flatten(state.latitude_deg),
        s=7,
        c="blue",
    )
    ax2.plot([-180, 180], [config.inclination_deg] * 2, "k--", linewidth=1)
    ax2.plot([-180, 180], [-config.inclination_deg] * 2, "k--", linewidth=1)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.set_xticks(np.arange(-180, 181, 30))
    ax2.set_yticks(np.arange(-90, 91, 15))
    ax2.set_xlabel("经度 [deg]")
    ax2.set_ylabel("地心纬度 [deg]")
    ax2.set_title(
        "Starlink Shell 1 星下点分布  |  "
        f"F = {config.phase_factor}  |  t = {state.time_s / 60.0:.2f} min"
    )
    ax2.grid(True)
    fig2.tight_layout()
    figures.append(("02_ground_points.png", fig2))

    # Figure 3: latitude histogram.
    fig3, ax3 = plt.subplots(figsize=(8.5, 4.5), facecolor="white")
    ax3.hist(
        state.matlab_flatten(state.latitude_deg),
        bins=np.arange(-90, 92, 2),
    )
    ax3.set_xlim(-90, 90)
    ax3.set_xlabel("地心纬度 [deg]")
    ax3.set_ylabel("卫星数量")
    ax3.set_title(
        "Starlink Shell 1 纬度分布  |  "
        f"{config.satellite_count}/{config.plane_count}/{config.phase_factor}"
    )
    ax3.grid(True)
    fig3.tight_layout()
    figures.append(("03_latitude_histogram.png", fig3))

    # Figure 4: Walker phase staggering.
    raan_degrees = np.rad2deg(
        np.repeat(state.raan0_rad, config.satellites_per_plane, axis=1)
    )
    u0_degrees = np.rad2deg(state.argument_of_latitude0_rad)
    fig4, ax4 = plt.subplots(figsize=(9.0, 5.5), facecolor="white")
    ax4.scatter(
        state.matlab_flatten(raan_degrees),
        state.matlab_flatten(u0_degrees),
        s=10,
    )
    ax4.set_xlim(0, 360)
    ax4.set_ylim(0, 360)
    ax4.set_xticks(np.arange(0, 361, 30))
    ax4.set_yticks(np.arange(0, 361, 30))
    ax4.set_xlabel("RAAN [deg]")
    ax4.set_ylabel(r"初始纬度幅角 $u_0$ [deg]")
    ax4.set_title(
        "Walker-Delta 相位结构："
        f"{config.satellite_count}/{config.plane_count}/{config.phase_factor}"
    )
    ax4.grid(True)
    fig4.tight_layout()
    figures.append(("04_walker_phase.png", fig4))

    return figures


def create_animation(config: ConstellationConfig, frame_count: int = 120) -> object:
    """Create one-orbit satellite animation and return ``FuncAnimation``."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    configure_matplotlib()
    raan0, u0 = initial_walker_geometry(config)
    omega_dot = raan_drift_rate_rad_s(config)

    figure = plt.figure(figsize=(9.0, 7.0), facecolor="white")
    axes = figure.add_subplot(111, projection="3d")
    draw_earth(axes)
    draw_orbit_planes(
        axes,
        config,
        raan0,
        color=(0.85, 0.85, 0.85),
        linewidth=0.25,
    )
    satellites = axes.scatter([], [], [], s=14, c="red", depthshade=False)
    style_3d_axes(axes, 1.10 * config.semi_major_axis_km)
    title = axes.set_title("")

    def update(frame_index: int) -> tuple[object, object]:
        current_time_s = frame_index / frame_count * config.orbital_period_s
        u = np.mod(u0 + config.mean_motion_rad_s * current_time_s, 2.0 * np.pi)
        raan = np.mod(raan0 + omega_dot * current_time_s, 2.0 * np.pi)
        x, y, z = eci_positions(
            config.semi_major_axis_km,
            config.inclination_rad,
            raan,
            u,
        )
        satellites._offsets3d = (
            x.ravel(order="F"),
            y.ravel(order="F"),
            z.ravel(order="F"),
        )
        title.set_text(
            "Starlink Shell 1  "
            f"{config.satellite_count}/{config.plane_count}/{config.phase_factor}"
            f"   t = {current_time_s / 60.0:.2f} min"
        )
        return satellites, title

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=30,
        blit=False,
        repeat=True,
    )
    figure.tight_layout()
    return animation


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="复现 Starlink Shell 1 的理想 Walker-Delta 星座。"
    )
    parser.add_argument(
        "-t",
        "--time-seconds",
        type=float,
        default=0.0,
        help="仿真时刻，单位为秒（默认：0）。",
    )
    parser.add_argument(
        "--no-j2",
        action="store_true",
        help="关闭 J2 引起的 RAAN 一阶长期漂移。",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="播放一个轨道周期的动画。",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="只计算并打印结果，不创建图形。",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不打开图形窗口，通常与 --save-dir 一起使用。",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="将四张静态图保存到指定目录。",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = ConstellationConfig(use_j2=not args.no_j2)
    state = propagate(config, args.time_seconds)
    print_report(config, state)

    if args.no_plots:
        return

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figures = create_static_plots(config, state)
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for filename, figure in figures:
            output_path = args.save_dir / filename
            figure.savefig(output_path, dpi=160, bbox_inches="tight")
            print(f"已保存图像：{output_path.resolve()}")

    animation = None
    if args.animate and not args.no_show:
        animation = create_animation(config)

    if args.no_show:
        plt.close("all")
    else:
        # Keep a live reference so Matplotlib does not garbage-collect animation.
        _ = animation
        plt.show()


if __name__ == "__main__":
    main()
