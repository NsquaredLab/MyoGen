raw_dir = '/Users/oj98yqyk/Downloads/raw';
out_dir = '/Users/oj98yqyk/code/MyoGen/data/experimental';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

files = dir(fullfile(raw_dir, 'Exp*_B_DATATABLE.mat'));
fprintf('Found %d files\n', length(files));

filepath = fullfile(files(1).folder, files(1).name);
fprintf('Loading %s...\n', files(1).name);
S = load(filepath);
dt = S.DATATABLE;

fprintf('Data class: %s\n', class(dt.Data));
if istable(dt.Data)
    fprintf('Table variables: %s\n', strjoin(dt.Data.Properties.VariableNames, ', '));
    fprintf('Table size: %d x %d\n', size(dt.Data));
    for v = 1:length(dt.Data.Properties.VariableNames)
        vname = dt.Data.Properties.VariableNames{v};
        val = dt.Data.(vname);
        if iscell(val)
            fprintf('  %s: cell [%s]', vname, mat2str(size(val)));
            if ~isempty(val) && ~isempty(val{1})
                fprintf(', first elem: %s [%s]', class(val{1}), mat2str(size(val{1})));
            end
            fprintf('\n');
        else
            fprintf('  %s: %s [%s]\n', vname, class(val), mat2str(size(val)));
        end
    end
end

fprintf('\nMUCorrs class: %s\n', class(dt.MUCorrs));
if istable(dt.MUCorrs)
    fprintf('MUCorrs variables: %s\n', strjoin(dt.MUCorrs.Properties.VariableNames, ', '));
end
