# Nonmonotone Adaptive Linesearch Algorithm (NALA) - MAIN branch

NALA python code (see [1]). The implementation uses python+pytorch

 NALA has been tested on the following Machine Learning applications:

## Problemi CUTEst

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
```
2. 86 problemi del lavoro [1]. I sif di questi problemi sono nella cartella `/home/giampo/SIF_FASANO`
> ```
'ARGLINA', 'ARGLINB', 'ARGLINC', 'ARWHEAD', 'BDQRTIC', 'BROWNAL',
'BRYBND', 'CHAINWOO', 'CLPLATEA_b', 'CLPLATEB', 'CLPLATEC',
'COSINE', 'CRAGGLVY', 'CURLY10', 'CURLY20', 'CURLY30_b', 'DIXMAANA_b',
'DIXMAANB_b', 'DIXMAANC_b', 'DIXMAAND_b', 'DIXMAANE_b', 'DIXMAANF_b',
'DIXMAANG_b', 'DIXMAANH_b', 'DIXMAANI_b', 'DIXMAANJ_b', 'DIXMAANK_b',
'DIXMAANL_b', 'DIXON3DQ', 'DQDRTIC', 'DQRTIC', 'EDENSCH', 'EG2',
'EIGENALS_b', 'EIGENBLS_b', 'EIGENCLS_b', 'ENGVAL1_b', 'EXTROSNB',
'FMINSRF2', 'FMINSURF_b', 'FREUROTH', 'GENROSE', 'HYDC20LS', 'LIARWHD',
'LMINSURF', 'MANCINO', 'MOREBV', 'MSQRTALS', 'MSQRTBLS', 'NCB20',
'NCB20B', 'NLMSURF', 'NONCVXU2', 'NONCVXUN', 'NONDIA', 'NONDQUAR',
'NONMSQRT_b', 'ODC', 'PENALTY1', 'PENALTY2', 'PENALTY3', 'POWELLSG',
'POWER_b', 'QUARTC', 'RAYBENDL', 'RAYBENDS', 'SBRYBND_b', 'SCHMVETT',
'SCOSINE', 'SCURLY10_b', 'SCURLY20_b', 'SCURLY30_b', 'SENSORS',
'SINQUAD_b', 'SPARSINE', 'SPARSQUR', 'SPMSRTLS_b', 'SROSENBR', 'SSC',
'TESTQUAD', 'TOINTGSS', 'TQUARTIC', 'TRIDIA', 'VARDIM', 'VAREIGVL', 'WOODS_b'
```


 ## Singoli problemi analitici

Attualmente sono implementati (nella cartella CODE_FOR) i seguenti problemi:

   * chebyquad7
   * dixon
   * oren
   * vardim
   * wood


 ## Problemi di Machine Learning

Attualmente è implementato il problema  `blogdata`


 ## AUTORI
 #### A. De Santis<sup>1</sup>, G. Liuzzi<sup>1</sup>, S. Lucidi<sup>1</sup>, E.M. Tronci<sup>1</sup>

 <sup>1</sup> Department of Computer, Control and Management Engineering, "Sapienza" University of Rome

 - desantis@diag.uniroma1.it,
 - liuzzi@diag.uniroma1.it,
 - lucidi@diag.uniroma1.it,
 - tronci@diag.uniroma1.it


 Copyright 2021

 ## References

[1] Fasano, G., & Lucidi, S. (2009). A nonmonotone truncated Newton–Krylov
     method exploiting negative curvature directions, for large scale unconstrained
     optimization. Optimization Letters, 3(4), 521-535.

[2] Curtis, F. E., & Robinson, D. P. (2019). Exploiting negative curvature
    in deterministic and stochastic optimization. Mathematical Programming,
    176(1), 69-94.
