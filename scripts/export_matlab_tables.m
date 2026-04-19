% export_matlab_tables.m
% Run this in MATLAB to export the DATATABLE .mat files to Python-readable format.
%
% Usage:
%   1. Open MATLAB
%   2. cd to the directory containing this script
%   3. Run: export_matlab_tables('/path/to/Downloads/raw')
%
% Output: Creates *_extracted.mat files in the same directory with:
%   - EMG: raw HD-sEMG signals (cell array of matrices, one per contraction)
%   - MUPulses: MU spike train indices (cell array of cell arrays)
%   - MUFilters: spatial filters / MUAP templates (if available)
%   - SIGlength: signal durations
%   - PNR: pulse-to-noise ratios
%   - fs: sampling frequency (if available in table properties)

function export_matlab_tables(raw_dir)
    if nargin < 1
        raw_dir = fullfile(getenv('HOME'), 'Downloads', 'raw');
    end

    files = dir(fullfile(raw_dir, 'Exp*_B_DATATABLE.mat'));
    fprintf('Found %d files in %s\n', length(files), raw_dir);

    for i = 1:length(files)
        filepath = fullfile(files(i).folder, files(i).name);
        fprintf('Processing %s...\n', files(i).name);

        try
            S = load(filepath);
            dt = S.DATATABLE;

            % Extract what we can from the table
            out = struct();

            % Try to get the actual data table contents
            if istable(dt.Data) || isa(dt.Data, 'table')
                varNames = dt.Data.Properties.VariableNames;
                fprintf('  Table variables: %s\n', strjoin(varNames, ', '));

                for v = 1:length(varNames)
                    out.(varNames{v}) = dt.Data.(varNames{v});
                end
            elseif iscell(dt.Data)
                % Cell array of data
                out.Data = dt.Data;
            else
                out.Data = dt.Data;
            end

            % Copy metadata fields
            fields_to_copy = {'MUCorrs', 'SUB', 'FoSoB', 'SpatFilt', ...
                              'DecBoundary', 'numMatches', 'numNotMatches', ...
                              'n_samples_EMG', 'SF_w_length', 'SIGlength', ...
                              'PNR', 'CoV_windows'};
            for f = 1:length(fields_to_copy)
                fname = fields_to_copy{f};
                if isfield(dt, fname)
                    out.(fname) = dt.(fname);
                end
            end

            % Save as v7 (readable by scipy)
            outpath = fullfile(raw_dir, strrep(files(i).name, '.mat', '_extracted.mat'));
            save(outpath, '-struct', 'out', '-v7');
            fprintf('  Saved to %s\n', outpath);

        catch ME
            fprintf('  ERROR: %s\n', ME.message);
        end
    end

    fprintf('Done. Export %d files.\n', length(files));
end
