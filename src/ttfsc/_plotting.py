from ._data_models import TTFSCResult


def plot_matplotlib(result: TTFSCResult, plot_matplotlib_style: str) -> None:
    from matplotlib import pyplot as plt

    plt.style.use(plot_matplotlib_style)
    plt.hlines(0, result.frequency_pixels[1], result.frequency_pixels[-2], "black")
    plt.plot(result.frequency_pixels[1:], result.fsc_values_unmasked[1:], label="FSC (unmasked)")
    if result.fsc_values_masked is not None:
        plt.plot(result.frequency_pixels[1:], result.fsc_values_masked[1:], label="FSC (masked)")
    if result.fsc_values_corrected is not None:
        plt.plot(result.frequency_pixels[1:], result.fsc_values_corrected[1:], label="FSC (corrected)")

    plt.xlabel("Resolution (Å)")
    plt.ylabel("Correlation")
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{(1 / x) * result.pixel_spacing_angstroms:.2f}")
    plt.xlim(result.frequency_pixels[1], result.frequency_pixels[-2])
    plt.ylim(-0.05, 1.05)
    plt.hlines(result.fsc_threshold, result.frequency_pixels[1], result.estimated_resolution_frequency_pixel, "red", "--")
    plt.vlines(result.estimated_resolution_frequency_pixel, -0.05, result.fsc_threshold, "red", "--")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_plottile(result: TTFSCResult) -> None:
    import plotille

    fig = plotille.Figure()
    fig.width = 60
    fig.height = 20
    fig.set_x_limits(float(result.frequency_pixels[1]), float(result.frequency_pixels[-1]))
    fig.set_y_limits(0, 1)
    fig.x_label = "Resolution [Å]"
    fig.y_label = "FSC"

    def resolution_callback(x: float, _: float) -> str:
        return f"{(1 / x) * result.pixel_spacing_angstroms:.2f}"

    fig.x_ticks_fkt = resolution_callback

    def fsc_callback(x: float, _: float) -> str:
        return f"{x:.2f}"

    fig.y_ticks_fkt = fsc_callback
    fig.plot(result.frequency_pixels[1:].numpy(), result.fsc_values_unmasked[1:].numpy(), lc="blue", label="FSC (unmasked)")
    if result.fsc_values_masked is not None:
        fig.plot(result.frequency_pixels[1:].numpy(), result.fsc_values_masked[1:].numpy(), lc="green", label="FSC (masked)")
    if result.fsc_values_corrected is not None:
        fig.plot(
            result.frequency_pixels[1:].numpy(),
            result.fsc_values_corrected[1:].numpy(),
            lc="yellow",
            label="FSC (corrected)",
        )
    fig.plot(
        [float(result.frequency_pixels[1].numpy()), result.estimated_resolution_frequency_pixel],
        [result.fsc_threshold, result.fsc_threshold],
        lc="red",
        label=" ",
    )
    fig.plot(
        [result.estimated_resolution_frequency_pixel, result.estimated_resolution_frequency_pixel],
        [0, result.fsc_threshold],
        lc="red",
        label=" ",
    )
    print(fig.show(legend=True))
