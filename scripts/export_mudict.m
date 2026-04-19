% Export MUdictionary data to Python-readable format
infile = '/Users/oj98yqyk/code/MyoGen/data/mudict/S16_50_KE.mat';
outfile = '/Users/oj98yqyk/code/MyoGen/data/mudict/S16_50_KE_extracted.mat';

S = load(infile);
fn = fieldnames(S);
fprintf('Top-level fields: %s\n', strjoin(fn, ', '));

% Explore structure
for i = 1:length(fn)
    val = S.(fn{i});
    fprintf('%s: %s [%s]\n', fn{i}, class(val), mat2str(size(val)));
    if isstruct(val)
        sf = fieldnames(val);
        fprintf('  subfields: %s\n', strjoin(sf, ', '));
        for j = 1:min(length(sf), 10)
            sv = val.(sf{j});
            if isnumeric(sv)
                fprintf('  .%s: %s [%s]\n', sf{j}, class(sv), mat2str(size(sv)));
            elseif iscell(sv)
                fprintf('  .%s: cell [%s]\n', sf{j}, mat2str(size(sv)));
                if ~isempty(sv) && ~isempty(sv{1})
                    fprintf('    {1}: %s [%s]\n', class(sv{1}), mat2str(size(sv{1})));
                end
            else
                fprintf('  .%s: %s\n', sf{j}, class(sv));
            end
        end
    end
end

% Extract key data and save as v7
out = struct();

% Try to find EMG data and MU pulse trains
if isfield(S, 'SIG')
    out.EMG = S.SIG;
    fprintf('\nEMG: %s [%s]\n', class(S.SIG), mat2str(size(S.SIG)));
end

if isfield(S, 'MUPulses')
    % Convert cell array to padded matrix
    n_mus = length(S.MUPulses);
    max_spikes = max(cellfun(@length, S.MUPulses));
    spike_matrix = nan(n_mus, max_spikes);
    for m = 1:n_mus
        spk = S.MUPulses{m};
        spike_matrix(m, 1:length(spk)) = spk(:)';
    end
    out.spike_trains = spike_matrix;
    out.n_mus = n_mus;
    fprintf('MU spike trains: %d MUs\n', n_mus);
end

if isfield(S, 'MUFilters')
    out.MUFilters = S.MUFilters;
end

if isfield(S, 'fsamp')
    out.fsamp = S.fsamp;
end

if isfield(S, 'edition')
    ed = S.edition;
    if isstruct(ed) && isfield(ed, 'MUPulses')
        n_mus_ed = length(ed.MUPulses);
        max_sp = max(cellfun(@length, ed.MUPulses));
        sp_mat = nan(n_mus_ed, max_sp);
        for m = 1:n_mus_ed
            spk = ed.MUPulses{m};
            sp_mat(m, 1:length(spk)) = spk(:)';
        end
        out.edited_spike_trains = sp_mat;
        out.n_mus_edited = n_mus_ed;
        fprintf('Edited MU spike trains: %d MUs\n', n_mus_ed);
    end
    if isstruct(ed) && isfield(ed, 'Pulsetrain')
        out.pulse_trains = ed.Pulsetrain;
    end
    if isstruct(ed) && isfield(ed, 'silval')
        out.sil_values = ed.silval;
    end
end

save(outfile, '-struct', 'out', '-v7');
fprintf('\nSaved to %s\n', outfile);
