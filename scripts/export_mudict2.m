infile = '/Users/oj98yqyk/code/MyoGen/data/mudict/S16_50_KE.mat';
outfile = '/Users/oj98yqyk/code/MyoGen/data/mudict/S16_50_KE_v2.mat';
S = load(infile);

out = struct();

% EMG data: signal.data [265 x 92416]
out.emg = S.signal.data;
out.fsamp = S.signal.fsamp;
out.n_channels = size(S.signal.data, 1);
out.n_samples = size(S.signal.data, 2);
fprintf('EMG: %d ch x %d samples at %d Hz\n', out.n_channels, out.n_samples, out.fsamp);

% Discharge times: signal.Dischargetimes [3 x 3] cell
% Flatten all MU discharge times into a list
dt = S.signal.Dischargetimes;
[nr, nc] = size(dt);
mu_count = 0;
all_spikes = {};
for r = 1:nr
    for c = 1:nc
        if ~isempty(dt{r,c})
            mu_count = mu_count + 1;
            all_spikes{mu_count} = dt{r,c}(:)';
            fprintf('MU %d (contr=%d, mu=%d): %d spikes\n', mu_count, r, c, length(dt{r,c}));
        end
    end
end

% Save spike trains as padded matrix
max_sp = max(cellfun(@length, all_spikes));
spike_matrix = nan(mu_count, max_sp);
for m = 1:mu_count
    spike_matrix(m, 1:length(all_spikes{m})) = all_spikes{m};
end
out.spike_trains = spike_matrix;
out.n_mus = mu_count;

% Force target
out.target = S.signal.target;
out.path = S.signal.path;

save(outfile, '-struct', 'out', '-v7');
fprintf('\nSaved %d MUs, %d channels to %s\n', mu_count, out.n_channels, outfile);
