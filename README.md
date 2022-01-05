# Adaptive Linesearch Algorithm (ALA)

ALA python code (see [1]). The implementation uses python+pytorch

3 versions of ALA have been tested:

- Nonmonotone-Monotone
- Monotone-Monotone
- Nonmonotone-Nonmonotone

ALA has been tested on the following Machine Learning applications:

## Machine Learning applications

1. 48 problemi del lavoro [2] (in cui evidenziano curvature negative). I sif dei problemi sono nella cartella `/home/giampo/sif`
> ```
'HEART8LS','OSBORNEB','LOGHAIRY','ECKERLE4LS','MISRA1ALS',
'DENSCHND','HEART6LS','BIGGS6','ROSZMAN1LS','NELSONLS','HAHN1LS','BENNETT5LS',
'MEYER3','MGH10LS','OSBORNEA','GROWTHLS','LANCZOS3LS','HUMPS','LANCZOS2LS',
'DENSCHNE','DANWOODLS','ENGVAL2','BEALE','ALLINITU',
'DJTL','ENSOLS','EXPFIT','HAIRY','KOWOSB','RAT43LS','SINEVAL',
'SNAIL','HATFLDE','HATFLDD','RAT42LS','DECONVU','GULF',
'HELIX', 'LANCZOS1LS', 'MGH09LS', 'POWELLBSLS', 'MGH17LS',
'THURBERLS', 'CHWIRUT1LS', 'CHWIRUT2LS', 'HYDC20LS',
'VIBRBEAM', 'KIRBY2LS'

 ## AUTORI
 #### A. De Santis<sup>1</sup>, G. Liuzzi<sup>1</sup>, S. Lucidi<sup>1</sup>, E.M. Tronci<sup>1</sup>

 <sup>1</sup> Department of Computer, Control and Management Engineering, "Sapienza" University of Rome

 - desantis@diag.uniroma1.it,
 - liuzzi@diag.uniroma1.it,
 - lucidi@diag.uniroma1.it,
 - tronci@diag.uniroma1.it


 Copyright 2022

 ## References

[1] Fasano, G., & Lucidi, S. (2009). A nonmonotone truncated Newton–Krylov
     method exploiting negative curvature directions, for large scale unconstrained
     optimization. Optimization Letters, 3(4), 521-535.
