function settings=gammaSGVBsettings(set)
    settings.initalpha=1;
    if set>0
    settings.useAdaGrad=0;
    settings.useAdadelta=0;
    settings.forgetting=1.0;
    settings.useAnalyticEntropy=1;
    settings.g2blend=.1;
    settings.msBlend=1;
    settings.stepSize=0.001;
    settings.samples=1000;
    settings.plot=0;
    settings.inita=[];
    settings.testGrad=1;
    settings.useMeanParameters=0;
    settings.numSamplesForL=10;
    end

    settings.samples=1000;
    settings.plot=0;
    settings.inita=[];
    settings.testGrad=1;
    settings.numSamplesForL=10;

    % msBlend=1 g2blend=1 is AdaGrad
    % msBlend=.9 g2blend=.1 is RMSprop
    switch set
        case 1 % sgd
            ;
        case 2 % + momentum
            settings.forgetting=.1;
        case 3 % RMSprop
            settings.useAdaGrad=1;
            settings.g2blend=.1;
            settings.msBlend=.9;
         case 4 % Adagrad
            settings.useAdaGrad=1;
            settings.g2blend=1;
            settings.msBlend=1;
        case 5 % adadelta
            settings.useAdadelta=1;
            settings.rho=.9;
            settings.eps=1e-4;
        case 6 % adadelta + mom
            settings.useAdadelta=1;
            settings.rho=.9;
            settings.eps=1e-4;
            settings.forgetting=.1;
        case 7 % use analytic gradient + momentum
            settings.forgetting=.1;
            settings.useAnalyticEntropy=1;
        case 8 % use mean parameters
            settings.forgetting=.1;
            settings.useMeanParameters=1;
        case 9 % adam
            settings.useAdadelta=0;
            settings.useAdaGrad=0;
            settings.useAdam=1;
	case -1
        settings.useAdadelta=0;
        settings.useAdaGrad=0;
        settings.useAdam=0;
	otherwise
	 error('do not support\n')
    end
end
