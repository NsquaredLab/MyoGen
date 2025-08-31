def spiketrainlists2binary(populations: list[sim.Population], save_path: str):
    """
    Saves the spike trains of the specified populations to a file.

    Parameters
    ----------
    populations : list[Population]
        The populations of neurons whose spike trains are to be saved.
    save_path : str
        The path where the spike trains will be saved.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the PyNN simulator is not running.
    """
    spike_trains = [
        population.get_data().segments[0].spiketrains for population in populations
    ]

    with open(save_path, "wb") as f:
        joblib.dump(spike_trains, f)
