import torch

def plot_matplotlib(
        fsc_values: torch.Tensor,
        resolution_angstroms: torch.Tensor,
        estimated_resolution_angstrom: float,
        fsc_threshold: float
        ):
    from matplotlib import pyplot as plt
    plt.hlines(0,resolution_angstroms[1], resolution_angstroms[-2],'black')
    plt.plot(resolution_angstroms, fsc_values,label="FSC (unmasked)")
    plt.xlabel('Resolution (Å)')
    plt.ylabel('Correlation')
    plt.xscale('log')
    plt.xlim(resolution_angstroms[1], resolution_angstroms[-2])
    plt.ylim(-0.05,1.05)
    plt.hlines(fsc_threshold,resolution_angstroms[1],estimated_resolution_angstrom,'red','--')
    plt.vlines(estimated_resolution_angstrom,-0.05,fsc_threshold,'red','--')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_plottile(
        fsc_values: torch.Tensor,
        frequency_pixels: torch.Tensor,
        pixel_spacing_angstroms: float,
        estimated_resolution_frequency_pixel: float,
        fsc_threshold: float
        ):
    import plotille

    fig = plotille.Figure()
    fig.width = 60
    fig.height = 20
    fig.set_x_limits(float(frequency_pixels[1]), float(frequency_pixels[-1]))
    fig.set_y_limits(0, 1)
    def resolution_callback(x,_):
        return '{:.2f}'.format((1 / x) * pixel_spacing_angstroms)
    fig.x_ticks_fkt = resolution_callback
    fig.plot(frequency_pixels[1:].numpy(), fsc_values[1:].numpy(),lc='blue',label='FSC')

    fig.plot([float(frequency_pixels[1].numpy()),estimated_resolution_frequency_pixel],[fsc_threshold,fsc_threshold],lc='red',label=' ')
    fig.plot([estimated_resolution_frequency_pixel,estimated_resolution_frequency_pixel],[0,fsc_threshold],lc='red',label=' ')
    print(fig.show(legend=True))