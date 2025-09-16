function eegStruct = loadEEGStruct(file_path)
    data = load(file_path, 'eegStruct');
    eegStruct = data.eegStruct;
end
