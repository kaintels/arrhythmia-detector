function [f_ECG] = filter_main(ECG, ecg_filter);


f_ECG = filtfilt(ecg_filter.Numerator, 1, ECG);
%fvtool(ecg_filter);
end

