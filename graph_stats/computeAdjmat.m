function A = computeAdjmat(X,pidx,params)
% computeAdjmat Compute adjacency matrix for sample-sample transcriptional
% similarity graph (correlation graph)
% INPUTS
%   X: nSamples x nProbes array of real-valued gene expression values
%   pidx: nProbes x 1 logical vector, indicates which probes to select
%   params: struct with fields
%       threshAwayNegative: boolean, if true, replace all negative values
%       by 0
% OUTPUTS
%   A: nSamples x nSamples adjacency matrix for a sample-sample transcriptional
%       similarity graph
% REQUIREMENTS
% - Stats toolbox
% VERSION
%   v1.0 JR

[A,~]=corr(X(:,pidx)');

% threshold away -ve values
if params.threshAwayNegative==true
    disp('thresholding away negative-valued edges');
    A=A.*double(A>0);
end

% resymmetrise to handle numerical imprecision
A=(A+A')/2;

% remove diag
A=A-diag(diag(A));
