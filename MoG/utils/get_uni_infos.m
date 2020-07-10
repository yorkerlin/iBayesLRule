function [uni_mean_info,uni_prec_info,uni_max_x, uni_min_x] = get_uni_infos(mixWeights,mixMeans,mixPrecs,offset)
    if nargin<4
        offset = 4;
    end
    assert(offset>0)
    d =length(mixMeans{1});
    nrComponents = length(mixWeights);
    mixVars = cell(nrComponents,1);

    uni_min_x = cell(d,1);
    uni_max_x = cell(d,1);
    uni_mean_info = cell(d,1);
    uni_prec_info = cell(d,1);

    for c=1:nrComponents
        mixVars{c} = inv(mixPrecs{c});
    end

    for k=1:d
        uni_mixMeans = cell(nrComponents,1);
        uni_mixPrecs = cell(nrComponents,1);
        mix_x = 0;
        max_x = 0;
        for c=1:nrComponents
            uni_mixMeans{c} = mixMeans{c}(k);
            uni_mixPrecs{c} = 1/ mixVars{c}(k,k);

            max_x_c =uni_mixMeans{c} + offset*sqrt(mixVars{c}(k,k));
            min_x_c =uni_mixMeans{c} - offset*sqrt(mixVars{c}(k,k));
            if c==1
                uni_max_x{k} = max_x_c;
                uni_min_x{k} = min_x_c;
            else
                uni_max_x{k} = max(max_x_c, uni_max_x{k});
                uni_min_x{k} = min(min_x_c, uni_min_x{k});
            end
        end
        uni_mean_info{k} = uni_mixMeans;
        uni_prec_info{k} = uni_mixPrecs;
    end


end
