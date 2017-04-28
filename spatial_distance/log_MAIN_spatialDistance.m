% This is code for analyses in our paper 'Distance is not everything in
% imaging genomics of functional networks: reply to a commentary on 
% _Correlated gene expression supports synchronous activity in brain
% networks_'
% Please refer to original paper
% Richiardi, Altmann et al., Science, 2015

%% Setup paths and data
disp(' *** SETTING UP *** ');

% setup paths
addpath(fullfile('..','graph_stats'));   % graph-based hypothesis testing
addpath(fullfile('..','library'));      % utility functions

% load gene expression data
load(fullfile('..','data','science_data.mat'));
% this contains normalised gene expression data on nProbes=58692 probes
% and nSamples=1777 samples, from the Allen Institute for Brain Science.
% This is processed for our needs, and we recommend downloading your own
% from <http://human.brain-map.org/static/download>.
%
% Variables:
%  * myX: nSamples x nProbes array of intercept-, batch- and subject- 
%   corrected microarray
% expression data. 
%  * myPredSel: nProbes x 1 vector of booleans indicating which probes to
%   use for each gene (see paper supplementary materials)
%  * prI: struct with probe information
%   'entrezID': Entrez Gene ID (NCBI, https://www.ncbi.nlm.nih.gov/gene)
%   'genesymbol': HUGO/HGNC accepted symbol (http://genenames.org/)
%  * cb: sample codebook with subject, coordinates, sample, and atlas
% information

% load tissue-tissue masks
% these contains a struct, tt, with the following fields
%   * tt_names  tissue class name for this sample
%   * tt_unique set of unique tissue class names
%   * TTmask 	nSamples x nSamples mask of edges linking two samples that
%               belong to the same tissue class
tt_f=load(fullfile('..','data','science_tt_fine.mat')); % fine tissue-tissue classes: 88
tt_c=load(fullfile('..','data','science_tt_coarse.mat')); % coarse tissue-tissue classes: 49
tt_f_v=jUpperTriMatToVec(tt_f.tt.TTmask,1); % vectorized version
tt_c_v=jUpperTriMatToVec(tt_c.tt.TTmask,1); % vectorized version

% compute adjacency matrices for sample-sample transcriptional similarity
params.threshAwayNegative=false;
A_pm=computeAdjmat(myX,myPredSel,params); % compute correlations using selected probes only
params.threshAwayNegative=true;
A_nn=computeAdjmat(myX,myPredSel,params); % same, no negative edges
A_nn_ttf = A_nn.*double(~tt_f.tt.TTmask); % also remove tissue-tissue edges (fine)
A_nn_ttc = A_nn.*double(~tt_c.tt.TTmask); % also remove tissue-tissue edges (coarse)

%% Euclidean distance and transcriptional similarity

% generate vectorized version for ease of reading
A_pm_v=jUpperTriMatToVec(A_pm,1);
A_nn_v=jUpperTriMatToVec(A_nn,1); % Transcriptional similarity between samples
A_nn_v_zidx=A_nn_v==0;  % indices for zeros
A_nn_ttf_v=jUpperTriMatToVec(A_nn_ttf,1); % Transcriptional similarity between samples
A_nn_ttf_v_zidx=A_nn_ttf_v==0; % indices for zeros
A_nn_ttc_v=jUpperTriMatToVec(A_nn_ttc,1); % Transcriptional similarity between samples
A_nn_ttc_v_zidx=A_nn_ttc_v==0; % indices for zeros

% pre-compute all vertex indices
nCommunities=15;
allVI=false(size(A_pm,1),nCommunities);
for cidx=1:nCommunities
    allVI(:,cidx)=cb.rcode==cidx;
end
% get within- and between- edges
c_all_nets=[1 3:14]; % 2 is basal ganglia, remove. 15 is rest of brain
c_within=[3 11 12 14]; % within-4-functional-networks communities. These are dDMN, salience, sensorimotor, visuospatial
wi_mask = logical(double(allVI(:,c_within))*double(allVI(:,c_within)')); % mask for within-4-functional-networks edges
wi_mask_v = jUpperTriMatToVec(wi_mask,1); % vector version of the above
w_mask = logical(double(allVI(:,c_all_nets))*double(allVI(:,c_all_nets)')); % mask for all functional network edges
w_mask_v = jUpperTriMatToVec(w_mask,1); % vector version of the above

% Compute Euclidean distances between samples
allDists=squareform(pdist(cb.center,'euclidean'));
allDists_v=jUpperTriMatToVec(allDists,1); 

% compute correlation between Euclidean distances and transcriptional
% similarity. This is to replicate Pantazatos and Li, and not the cleanest
% thing to do because subject random effects (clustered data) are ignored.

[corr_nn,corr_nn_p]=corr(allDists_v,A_nn_v)
% only look at non-negative edges
[corr_nn_nz,corr_nn_nz_p]=corr(allDists_v(~A_nn_v_zidx),A_nn_v(~A_nn_v_zidx))
sum(~A_nn_v_zidx)-2 % degrees of freedom
% remove fine tissue-tissue correlations
[corr_nn_ttf_nz,corr_nn_ttf_nz_p]=corr(allDists_v(~A_nn_ttf_v_zidx),A_nn_v(~A_nn_ttf_v_zidx))
sum(~A_nn_ttf_v_zidx)-2 % dof
% remove coarse tissue-tissue correlations
[corr_nn_ttc_nz,corr_nn_ttc_nz_p]=corr(allDists_v(~A_nn_ttc_v_zidx),A_nn_v(~A_nn_ttc_v_zidx))
sum(~A_nn_ttc_v_zidx)-2 % dof


dists_wi_v=allDists_v(wi_mask_v); % vector of all within-4-functional-networks edge distances
dists_wi_ttf_nz_v=allDists_v(logical(wi_mask_v.*~tt_f_v)); % same, but removing same-tissue (fine) edges
dists_wi_ttc_nz_v=allDists_v(logical(wi_mask_v.*~tt_c_v)); % same, but removing same-tissue (coarse) edges
dists_tmw_v=allDists_v(~w_mask_v); % vector of all edges that are not within-any-functional-network (T-W edges)


%% regressing out Euclidean distance
% model 1: regress on full matrix (fix rank deficiency by pre-scaling)
my_model_pm=fitlm(allDists_v/100,A_pm_v)
% distance-regressed vector of transcriptional similarities = prediction residuals
allTS_clean_pm_v=my_model_pm.Residuals.Raw; 

% likewise, model 2: regress on non-neg corrs only
my_model_nn = fitlm(allDists_v(~A_nn_v_zidx)/100,A_nn_v(~A_nn_v_zidx))

% likewise, model 3: regress on non-neg, non-tissue-tissue (fine) corrs only
my_model_nn_tt = fitlm(allDists_v(~A_nn_ttf_v_zidx)/100,A_nn_v(~A_nn_ttf_v_zidx))


% rest of code coming soon