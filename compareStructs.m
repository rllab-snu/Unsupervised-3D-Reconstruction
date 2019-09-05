%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code from http://cvlab.lums.edu.pk/nrsfm/
% Ijaz Akhter, Yaser Sheikh, Sohaib Khan, Takeo Kanade,
%  "Nonrigid Structure from Motion in Trajectory Space",
%  in Proceedings of 22nd Annual Conference on Neural Information Processing Systems, NIPS 2008, Vancouver, Canada, pp. 41-48, December 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function errS = compareStructs(S, Shat)

s = mean(std(S, 1, 2));
S = S/s;
sm = mean(S,2);
S = S - sm*ones(1,size(S,2));

Shat = Shat/s;
F = size(S,1)/3;
P = size(S,2);

for i=1:F
    errS(i)  = sum(sqrt( sum( (S(3*i-2:3*i, :)-Shat(3*i-2:3*i, :)).^2) ) )/P;
end