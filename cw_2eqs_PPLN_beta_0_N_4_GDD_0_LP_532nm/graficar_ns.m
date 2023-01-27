clc; clear all; close all;
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');  
set(0,'defaultAxesFontSize',20)

c  = 299792458*1e6/1e12; % speed of light in vac. [μm/ps]

PG = load('ParamGain.dat');

% FOLDER = {
%     'ns_2eqs_PPLN_delta_0_POWER_75W',...    
%     'ns_2eqs_PPLN_delta_0_POWER_100W',...
%     'ns_2eqs_PPLN_delta_0_POWER_250W',...
%     };

FOLDER = {
    'ns_2eqs_PPLN_delta_-0.5_POWER_100W',...
    };

h = figure('units','normalized','outerposition',[0 0 1 1]);

for fd = 1:length(FOLDER)

    folder = FOLDER{fd};
        
    T    = load([folder,'/T.dat']);
    trt  = T(end)-T(1); % ps
    Tp   = load([folder,'/Tp.dat']);
    f    = load([folder,'/freq.dat']);
    
    
    signal_r=load([folder,'/signal_output_r.dat']);
    signal_i=load([folder,'/signal_output_i.dat']);

    pump_r=load([folder,'/pump_output_r.dat']);
    pump_i=load([folder,'/pump_output_i.dat']);
    pump_input=load([folder,'/pump_input_r.dat']);


    SIGNAL  = signal_r + 1j*signal_i;
    PUMP   = pump_r  + 1j*pump_i;

    idler = false;
    if(idler)
        dips('Idler included');
        idler_r = load([folder,'/idler_output_r.dat']);
        idler_i = load([folder,'/idler_output_i.dat']); 
        IDLER  = idler_r + 1j*idler_i;
    %     IDLERW = ifftshift(ifft(IDLER));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    C    = 299792458*1e6/1e12; % speed of ligth in vacuum [um/ps]
    EPS0 = 8.8541878128e-12*1e12/1e6; %vacuum pertivity [W.ps/V²μm] 
    np = 2.22515;	
    ns = 2.14883; 
    ni= ns;
    
    waist       = 55; % beam waist radius [um]
    spot        = pi*waist^2; % spot area [μm²]
    
    cp = .5 * EPS0 * C * spot * np;
    cs = .5 * EPS0 * C * spot * ns * sqrt(2);
    ci = .5 * EPS0 * C * spot * ni * sqrt(2);

    rts = 600;
    x      = 1:rts;
    Is     = cs*abs(SIGNAL).^2;
    Ii     = ci*abs(SIGNAL).^2;
    Ip     = cp*abs(PUMP).^2;
    Ip_in     = cp*abs(pump_input).^2;

    pmeds     = zeros(1,rts);
    pmedi     = zeros(1,rts);
    pmedp     = zeros(1,rts);
    pmedp_in  = zeros(1,rts);
    for i =1:rts
        pmeds(i)     = trapz(T,Is(((i-1)*length(T)+1):(i*length(T))))/trt;
        pmedi(i)     = trapz(T,Ii(((i-1)*length(T)+1):(i*length(T))))/trt;
        pmedp(i)     = trapz(T,Ip(((i-1)*length(T)+1):(i*length(T))))/trt;
        pmedp_in(i)  = trapz(T,Ip_in(((i-1)*length(T)+1):(i*length(T))))/trt;
    end

    d_time = (x-x(end/2))*trt*1e-3;
    subplot(length(FOLDER),1, fd)
    yyaxis left
    hold on
    plot( d_time, pmedp_in )
    plot( d_time, pmedp )
    ylabel('Pump power (W)')
    yyaxis right
    plot( d_time, pmeds*0.02 )
    plot( d_time, pmedi*0.02 )
    xlabel('Time (ns)')
    ylabel('Signal power (W)')
    ax= gca; ax.PlotBoxAspectRatio = [2,1,1];
    box on; grid on;
%     legend({'$\langle P_p^{\mathrm{in}}\rangle_{\mathrm{t}_{\mathrm{rt}}}$',...
%         '$\langle P_p^{\mathrm{out}}\rangle_{\mathrm{t}_{\mathrm{rt}}}$',...
%         '$\langle P_s^{\mathrm{out}}\rangle_{\mathrm{t}_{\mathrm{rt}}}$'},...
%         'Interpreter', 'latex')
end

SIGNALW = ifftshift(ifft(SIGNAL));
figure();
hold on
area(PG(:,1),PG(:,2)/max(PG(:,2))*8e-3, 'FaceAlpha', 0.2)
plot(f, abs(SIGNALW).^2)