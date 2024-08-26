from typing import Optional

import torch


def plot_matplotlib(
    fsc_values_unmasked: torch.Tensor,
    fsc_values_masked: Optional[torch.Tensor],
    fsc_values_corrected: Optional[torch.Tensor],
    frequency_pixels: torch.Tensor,
    pixel_spacing_angstroms: float,
    estimated_resolution_frequency_pixel: float,
    fsc_threshold: float,
    plot_matplotlib_style: str,
) -> None:
    from matplotlib import pyplot as plt

    plt.style.use(plot_matplotlib_style)
    plt.hlines(0, frequency_pixels[1], frequency_pixels[-2], "black")
    plt.plot(frequency_pixels[1:], fsc_values_unmasked[1:], label="FSC (unmasked)")
    if fsc_values_masked is not None:
        plt.plot(frequency_pixels[1:], fsc_values_masked[1:], label="FSC (masked)")
    if fsc_values_corrected is not None:
        plt.plot(frequency_pixels[1:], fsc_values_corrected[1:], label="FSC (corrected)")

    plt.xlabel("Resolution (Å)")
    plt.ylabel("Correlation")
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{(1 / x) * pixel_spacing_angstroms:.2f}")
    plt.xlim(frequency_pixels[1], frequency_pixels[-2])
    plt.ylim(-0.05, 1.05)
    plt.hlines(fsc_threshold, frequency_pixels[1], estimated_resolution_frequency_pixel, "red", "--")
    plt.vlines(estimated_resolution_frequency_pixel, -0.05, fsc_threshold, "red", "--")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_plottile(
    fsc_values: torch.Tensor,
    fsc_values_masked: Optional[torch.Tensor],
    fsc_values_corrected: Optional[torch.Tensor],
    frequency_pixels: torch.Tensor,
    pixel_spacing_angstroms: float,
    estimated_resolution_frequency_pixel: float,
    fsc_threshold: float,
) -> None:
    import plotille

    fig = plotille.Figure()
    fig.width = 60
    fig.height = 20
    fig.set_x_limits(float(frequency_pixels[1]), float(frequency_pixels[-1]))
    fig.set_y_limits(0, 1)
    fig.x_label = "Resolution [Å]"
    fig.y_label = "FSC"

    def resolution_callback(x: float, _: float) -> str:
        return f"{(1 / x) * pixel_spacing_angstroms:.2f}"

    fig.x_ticks_fkt = resolution_callback

    def fsc_callback(x: float, _: float) -> str:
        return f"{x:.2f}"

    fig.y_ticks_fkt = fsc_callback
    fig.plot(frequency_pixels[1:].numpy(), fsc_values[1:].numpy(), lc="blue", label="FSC (unmasked)")
    if fsc_values_masked is not None:
        fig.plot(frequency_pixels[1:].numpy(), fsc_values_masked[1:].numpy(), lc="green", label="FSC (masked)")
    if fsc_values_corrected is not None:
        fig.plot(frequency_pixels[1:].numpy(), fsc_values_corrected[1:].numpy(), lc="yellow", label="FSC (corrected)")
    fig.plot(
        [float(frequency_pixels[1].numpy()), estimated_resolution_frequency_pixel],
        [fsc_threshold, fsc_threshold],
        lc="red",
        label=" ",
    )
    fig.plot(
        [estimated_resolution_frequency_pixel, estimated_resolution_frequency_pixel], [0, fsc_threshold], lc="red", label=" "
    )
    print(fig.show(legend=True))
