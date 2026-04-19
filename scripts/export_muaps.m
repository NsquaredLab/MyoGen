out_dir = '/Users/oj98yqyk/code/MyoGen/data/experimental';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

raw_dir = '/Users/oj98yqyk/Downloads/raw';
files = dir(fullfile(raw_dir, 'Exp*_B_DATATABLE.mat'));
fprintf('Exporting %d experiments...\n', length(files));

for fi = 1:length(files)
    filepath = fullfile(files(fi).folder, files(fi).name);
    fprintf('Processing %s...\n', files(fi).name);

    S = load(filepath);
    dt = S.DATATABLE;
    T = dt.Data;

    n_contractions = size(T, 1);
    n_mu_cols = size(T.MU, 2);

    % Extract experiment number from filename
    tokens = regexp(files(fi).name, 'Exp(\d+)', 'tokens');
    exp_num = str2double(tokens{1}{1});

    % For each contraction, extract MU spike trains and spatial filters
    for ci = 1:n_contractions
        % Collect non-empty MUs for this contraction
        mu_spikes = {};
        mu_sf = {};
        for mi = 1:n_mu_cols
            spk = T.MU{ci, mi};
            sf = T.SF{ci, mi};
            if ~isempty(spk) && numel(spk) > 2
                mu_spikes{end+1} = spk(:)';
                mu_sf{end+1} = sf(:)';
            end
        end

        n_mus = length(mu_spikes);
        if n_mus == 0, continue; end

        % Get task name
        taskname = T.Taskname(ci);
        if isstring(taskname), taskname = char(taskname); end

        % Get MUPulses if available
        mupulses = {};
        if ci <= size(T.MUPulses, 1) && ~isempty(T.MUPulses{ci})
            mp = T.MUPulses{ci};
            if iscell(mp)
                for pi = 1:length(mp)
                    if ~isempty(mp{pi})
                        mupulses{end+1} = mp{pi}(:)';
                    end
                end
            end
        end

        % Save as v7 .mat (scipy-readable)
        outfile = fullfile(out_dir, sprintf('exp%02d_contr%02d.mat', exp_num, ci));

        % Convert cell arrays to struct arrays for scipy compatibility
        out = struct();
        out.exp_num = exp_num;
        out.contraction = ci;
        out.taskname = taskname;
        out.n_mus = n_mus;
        out.n_samples = dt.n_samples_EMG(ci);
        out.sig_length_s = dt.SIGlength(ci);

        % Save spike trains as padded matrix (n_mus x max_spikes), NaN-padded
        max_spikes = max(cellfun(@length, mu_spikes));
        spike_matrix = nan(n_mus, max_spikes);
        for mi = 1:n_mus
            spike_matrix(mi, 1:length(mu_spikes{mi})) = mu_spikes{mi};
        end
        out.spike_trains = spike_matrix;

        % Save spatial filters as matrix (n_mus x n_channels)
        n_ch = length(mu_sf{1});
        sf_matrix = zeros(n_mus, n_ch);
        for mi = 1:n_mus
            sf_matrix(mi, 1:min(length(mu_sf{mi}), n_ch)) = mu_sf{mi}(1:min(length(mu_sf{mi}), n_ch));
        end
        out.spatial_filters = sf_matrix;

        % PNR for this contraction
        if ci <= length(dt.PNR) && ~isempty(dt.PNR{ci})
            out.pnr = dt.PNR{ci}(:)';
        end

        % MUPulses
        if ~isempty(mupulses)
            max_pulses = max(cellfun(@length, mupulses));
            pulse_matrix = nan(length(mupulses), max_pulses);
            for pi = 1:length(mupulses)
                pulse_matrix(pi, 1:length(mupulses{pi})) = mupulses{pi};
            end
            out.mu_pulses = pulse_matrix;
        end

        save(outfile, '-struct', 'out', '-v7');
        fprintf('  Contraction %d: %d MUs, %d samples -> %s\n', ci, n_mus, out.n_samples, outfile);
    end
end

fprintf('Done.\n');
