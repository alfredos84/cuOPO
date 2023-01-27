clc; clear all; close all;
set(0,'defaulttextinterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');  
set(0,'defaultAxesFontSize',20)


folder = 'cw_3eqs_PPLN_delta_0_POWER_0.5';
T      = load([folder,'/T.dat']);
trt    = T(end)-T(1); % ps
Tp     = load([folder,'/Tp.dat']);
Tp     = Tp + max(Tp);
f      = load([folder,'/freq.dat']);


h = figure('units','normalized','outerposition',[0 0 1 1]);

powers = [0.5 0.75 0.9 0.95 0.98 1 1.02 1.05 1.1 1.25 1.5 2 2.5 3 3.5 4 4.5];% 5 5.5 6];
pmeds     = zeros(1,length(powers));
pmedp     = zeros(1,length(powers));
avs       = zeros(1,length(powers));
avp       = zeros(1,length(powers));
pmedp_in  = zeros(1,length(powers));

i = 1;
for P = powers

    folder = ['cw_3eqs_PPLN_delta_0_POWER_',num2str(P)]    
    
    signal_r=load([folder,'/signal_output_r.dat']);
    signal_i=load([folder,'/signal_output_i.dat']);

    pump_r=load([folder,'/pump_output_r.dat']);
    pump_i=load([folder,'/pump_output_i.dat']);
    pump_input=load([folder,'/pump_input_r.dat']);

    SIGNAL  = signal_r + 1j*signal_i;
    PUMP   = pump_r  + 1j*pump_i;
    
    idler = true;
    if(idler)
        idler_r = load([folder,'/idler_output_r.dat']);
        idler_i = load([folder,'/idler_output_i.dat']); 
        IDLER  = idler_r + 1j*idler_i;
    %     IDLERW = ifftshift(ifft(IDLER));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    C    = 299792458*1e6/1e12; % speed of ligth in vacuum [um/ps]
    EPS0 = 8.8541878128e-12*1e12/1e6; %vacuum pertivity [W.ps/V²μm] 
    np = 2.22515;    ns = 2.14883;     ni= ns;

    waist       = 55; % beam waist radius [um]
    spot        = pi*waist^2; % spot area [μm²]

    cp = .5 * EPS0 * C * spot * np;
    cs = .5 * EPS0 * C * spot * ns * sqrt(2);
    ci = .5 * EPS0 * C * spot * ni * sqrt(2);

    Ps    = cs*abs(SIGNAL).^2;
    Pi    = ci*abs(IDLER).^2;
    Pp    = cp*abs(PUMP).^2;
    Pp_in = cp*abs(pump_input).^2;

    pmeds(i)  = trapz(Tp,Ps)/(Tp(end)-Tp(1));
    pmedi(i)  = trapz(Tp,Pi)/(Tp(end)-Tp(1));
	pmedp(i)  = trapz(Tp,Pp)/(Tp(end)-Tp(1));
    avs(i)    = mean(abs(SIGNAL).^2);
    avi(i)    = mean(abs(IDLER).^2);
    avp(i)    = mean(abs(PUMP).^2);
    
    i = i+1;
end


Pth      = 2.72562003135681;
Pth_new  = Pth;%2*2.72562003135681/(0.98+0.6);

yyaxis left
hold on
plot( sqrt(powers*Pth_new/Pth), pmeds*0.02 )
plot( sqrt(powers*Pth_new/Pth), pmedi*0.02 )
ylabel('Output signal power (W)')
yyaxis right
plot( sqrt(powers*Pth_new/Pth), 1-pmedp/mean(Pp_in))
xlabel('Input pump power (W)')
ylabel('Output pump power (W)')
ax= gca; ax.PlotBoxAspectRatio = [2,1,1];
box on; grid on;