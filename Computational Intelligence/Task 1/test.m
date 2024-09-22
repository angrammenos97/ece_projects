K = [1 2 5 10 15 30 50];
for i=1:length(K)
    in(i) = Simulink.SimulationInput('Control_Design');
    in(i) = in(i).setBlockParameter('Control_Design/Fuzzy PI Controller/K', 'Gain', num2str(K(i)));
end

out = parsim(in,'ShowSimulationManager','on','ShowProgress','on', 'TransferBaseWorkspaceVariables','on');

% openExample('simulink_general/sldemo_househeatExample');
% open_system('sldemo_househeat');
% SetPointValues = 65:2:85;
% spv_length = length(SetPointValues);
% for i = spv_length:-1:1
%     in(i) = Simulink.SimulationInput('sldemo_househeat');
%     in(i) = in(i).setBlockParameter('sldemo_househeat/Set Point',...
%         'Value',num2str(SetPointValues(i)));
% end
% out = parsim(in,'ShowSimulationManager','on','ShowProgress','on')