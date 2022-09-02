clc; clear all; %close all;
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


% h = figure('units','normalized','outerposition',[0 0 1 1]);

powers = [0.5 0.75 0.9 0.95 0.98 1 1.02 1.05 1.1 1.25 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6];
pmeds     = zeros(1,length(powers));
pmedi     = zeros(1,length(powers));
pmedp     = zeros(1,length(powers));

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
    cs = .5 * EPS0 * C * spot * ns;% * sqrt(2);
    ci = .5 * EPS0 * C * spot * ni;% * sqrt(2);

    Ps    = cs*abs(SIGNAL(end-length(T)+1:end)).^2;
    Pi    = ci*abs(IDLER(end-length(T)+1:end)).^2;
    Pp    = cp*abs(PUMP(end-length(T)+1:end)).^2;
    Pp_in = cp*abs(pump_input).^2;

    pmeds(i)    = trapz(T,Ps)/trt;
    pmedi(i)    = trapz(T,Pi)/trt;
	pmedp(i)    = trapz(T,Pp)/trt;
   
    pmedp_in(i) = trapz(T,Pp_in)/trt;
    
    i = i+1;
end

subplot(2,1,2)
hold on
plot( sqrt(powers), pmeds*(1-.98) )
plot( sqrt(powers), pmedi*(1-.6) )
xlabel('$\sqrt{N}$ (arbit. units)')
ylabel('Output signal power (W)')
ax= gca; ax.PlotBoxAspectRatio = [2,1,1];
box on; grid on;
legend({'$\lambda_s=1060$ nm', '$\lambda_i=1068$ nm' }, 'Interpreter', 'latex')
% 
% subplot(2,2,4)
% yyaxis left
% hold on
% plot( (powers), pmeds./pmedp/max(pmeds./pmedp))
% plot( (powers), pmedi./pmedp/max(pmedi./pmedp))
% ylabel('Norm. conv. effic. , $\eta$')
% yyaxis right
% hold on
% plot( (powers), (1-pmedp./pmedp_in)*100)
% xlabel('$N$ (arbit. units)')
% ylabel('Pump deplection (\%)')
% ax= gca; ax.PlotBoxAspectRatio = [1,1,1];
% box on; grid on;
% hold on; 
