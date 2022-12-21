% 初始化连接
stkInit;
% 默认端口 5001
remMachine = stkDefaultHost;
% conid只能获取两次，使用完需要关闭连接
conid = stkOpen(remMachine);
objNames = stkObjNames;
dt = 1;
style = 'Access';
startTime = 0;
endTime = 1;
result = zeros(8,50);
plane=8;
numperplane=9;
for i = 1:7 %平面内的参数
%     disp(i);
    for j = 1:9
%         disp(j);
%         if j<10 
%             j=strcat('0',num2str(j));
%         else
%             j=num2str(j);
%         end
        src=strcat('/Scenario/LoadTest/Satellite/LEO',num2str(i),num2str(j));
        dst=strcat('/Scenario/LoadTest/Satellite/LEO',num2str(i+1),num2str(j));
        disp(strcat(src,'->',dst));
%         [secData, secNames] =stkAccReport(src,dst, style);
        [secData, ~] =stkAccReport(src,dst, 'AER',0,86400,10);
        myTable = struct2table(secData{1});
        fileName = strcat('D:\STK_MATLAB\matlab_Code\72-d',extractAfter(src,"LEO"),'-',extractAfter(dst,"LEO"),'.csv');
        writetable(myTable,fileName,'delimiter',',');
    end
end
for i = 1:8 %平面内的参数
%     disp(i);
    for j = 1:9
%         disp(j);
%         if j<10 
%             j=strcat('0',num2str(j));
%         else
%             j=num2str(j);
%         end
        src=strcat('/Scenario/LoadTest/Satellite/LEO',num2str(i),num2str(j));
        disp(src);
        [secData, secNames] =stkReport(strcat('/Scenario/LoadTest/Satellite/LEO',num2str(i),num2str(j)), 'LLA Position',0,86400,1);
        myTable = struct2table(secData{1});
        fileName = strcat('./72-loads/',extractAfter(src,"LEO"),'.csv');
        writetable(myTable,fileName,'delimiter',',');
    end
end
src=strcat('/Scenario/LoadTest/Satellite/LEO','1','1');
dst=strcat('/Scenario/LoadTest/Satellite/LEO','2','1');
disp(strcat(src,'->',dst));
[secData, ~] =stkAccReport(src,dst, 'AER',0,86400,10);
[secData, secNames] =stkReport(strcat('/Scenario/LoadTest/Satellite/LEO','1','1'), 'LLA Position',0,86400,1);
stkClose(conid);