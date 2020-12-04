[![Work in Repl.it](https://classroom.github.com/assets/work-in-replit-14baed9a392b3a25080506f3b7b6d57f295ec2978f6f33ec97e36a161684cbe9.svg)](https://classroom.github.com/online_ide?assignment_repo_id=3659717&assignment_repo_type=AssignmentRepo)
# MiniTorch Module 4

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

* Pictures and Outputs for Task 4.5

    *  1 picture from the Visdom plot while training
        ![Simple Graph](https://github.com/Cornell-Tech-ML/minitorch-4-runefeather/blob/master/single_picture.png)
    * A screenshot of the Visdom screen while training
        ![Simple Loss](https://github.com/Cornell-Tech-ML/minitorch-4-runefeather/blob/master/screenshot.png)
    *   Output for training
        ```
        Epoch  0  example  0  loss  36.70911683802039  accuracy  0.125
        Epoch  0  example  800  loss  1840.0362307193643  accuracy  0.15
        Epoch  0  example  1600  loss  1824.6626054101362  accuracy  0.2
        Epoch  0  example  2400  loss  1510.273205200656  accuracy  0.6125
        Epoch  0  example  3200  loss  1052.5481905542215  accuracy  0.7375
        Epoch  0  example  4000  loss  842.8846101669819  accuracy  0.6375
        Epoch  0  example  4800  loss  887.3068812568001  accuracy  0.65
        Epoch  1  example  0  loss  16.098653479560706  accuracy  0.75
        Epoch  1  example  800  loss  775.0309906748806  accuracy  0.7375
        Epoch  1  example  1600  loss  822.0102865122822  accuracy  0.675
        Epoch  1  example  2400  loss  552.4138050372713  accuracy  0.75
        Epoch  1  example  3200  loss  621.2881501485588  accuracy  0.7125
        Epoch  1  example  4000  loss  547.3953986354848  accuracy  0.7625
        Epoch  1  example  4800  loss  540.8835167218152  accuracy  0.775
        Epoch  2  example  0  loss  6.284182104603355  accuracy  0.7625
        Epoch  2  example  800  loss  531.3824788212073  accuracy  0.7625
        Epoch  2  example  1600  loss  556.6554726006402  accuracy  0.775
        Epoch  2  example  2400  loss  404.96639142654226  accuracy  0.7875
        Epoch  2  example  3200  loss  406.17473347584604  accuracy  0.8
        Epoch  2  example  4000  loss  425.0093997716939  accuracy  0.7875
        Epoch  2  example  4800  loss  361.47946792306175  accuracy  0.775
        Epoch  3  example  0  loss  3.3654386962359664  accuracy  0.85
        Epoch  3  example  800  loss  364.1653534923177  accuracy  0.8375
        Epoch  3  example  1600  loss  459.9633948543721  accuracy  0.8
        Epoch  3  example  2400  loss  301.70018045434875  accuracy  0.8125
        Epoch  3  example  3200  loss  326.36201362522763  accuracy  0.8625
        Epoch  3  example  4000  loss  309.07314597557763  accuracy  0.8375
        Epoch  3  example  4800  loss  315.4938282536802  accuracy  0.775
        Epoch  4  example  0  loss  4.322821656527174  accuracy  0.825
        Epoch  4  example  800  loss  292.15093785907857  accuracy  0.8375
        Epoch  4  example  1600  loss  407.2079739211289  accuracy  0.775
        Epoch  4  example  2400  loss  286.3925231457834  accuracy  0.825
        Epoch  4  example  3200  loss  293.35869688006983  accuracy  0.875
        Epoch  4  example  4000  loss  271.88312508955187  accuracy  0.875
        Epoch  4  example  4800  loss  260.95754033842405  accuracy  0.8375
        Epoch  5  example  0  loss  1.5569251389315095  accuracy  0.825
        Epoch  5  example  800  loss  258.88654597764696  accuracy  0.85
        Epoch  5  example  1600  loss  355.2606182285952  accuracy  0.825
        Epoch  5  example  2400  loss  269.6540286150091  accuracy  0.775
        Epoch  5  example  3200  loss  284.32826193896346  accuracy  0.8375
        Epoch  5  example  4000  loss  263.17436423059985  accuracy  0.8125
        Epoch  5  example  4800  loss  273.8791621361427  accuracy  0.825
        Epoch  6  example  0  loss  2.211568695994912  accuracy  0.8375
        Epoch  6  example  800  loss  234.94641579682659  accuracy  0.8625
        Epoch  6  example  1600  loss  329.97265870038564  accuracy  0.875
        Epoch  6  example  2400  loss  229.284472803265  accuracy  0.7875
        Epoch  6  example  3200  loss  233.34489773436542  accuracy  0.8625
        Epoch  6  example  4000  loss  254.85724414175647  accuracy  0.875
        Epoch  6  example  4800  loss  216.73727531059077  accuracy  0.8625
        Epoch  7  example  0  loss  2.829143330033658  accuracy  0.9125
        Epoch  7  example  800  loss  229.18090816536858  accuracy  0.825
        Epoch  7  example  1600  loss  309.23706560534777  accuracy  0.8375
        Epoch  7  example  2400  loss  247.00494088677996  accuracy  0.8125
        Epoch  7  example  3200  loss  245.80956914155703  accuracy  0.8875
        Epoch  7  example  4000  loss  242.85773226891953  accuracy  0.85
        Epoch  7  example  4800  loss  273.6755159478107  accuracy  0.8
        Epoch  8  example  0  loss  4.869117252098119  accuracy  0.875
        Epoch  8  example  800  loss  288.35093449691846  accuracy  0.8625
        Epoch  8  example  1600  loss  328.06355876053595  accuracy  0.85
        Epoch  8  example  2400  loss  217.9287838319383  accuracy  0.8125
        Epoch  8  example  3200  loss  240.74581167318772  accuracy  0.8625
        Epoch  8  example  4000  loss  236.82408802781276  accuracy  0.8
        Epoch  8  example  4800  loss  212.31324577276888  accuracy  0.85
        Epoch  9  example  0  loss  4.401990359782494  accuracy  0.8125
        Epoch  9  example  800  loss  234.80722425047156  accuracy  0.8625
        Epoch  9  example  1600  loss  299.7560318689434  accuracy  0.85
        Epoch  9  example  2400  loss  179.21920548903418  accuracy  0.825
        Epoch  9  example  3200  loss  205.66182152869425  accuracy  0.825
        Epoch  9  example  4000  loss  177.6496294098172  accuracy  0.875
        Epoch  9  example  4800  loss  214.67766605472616  accuracy  0.8625
        Epoch  10  example  0  loss  4.663431757163864  accuracy  0.8875
        Epoch  10  example  800  loss  209.82375329921325  accuracy  0.8625
        Epoch  10  example  1600  loss  309.43265161500017  accuracy  0.8625
        Epoch  10  example  2400  loss  211.15840497543826  accuracy  0.925
        Epoch  10  example  3200  loss  203.26778077611147  accuracy  0.9
        Epoch  10  example  4000  loss  167.65422238003444  accuracy  0.8875
        Epoch  10  example  4800  loss  196.08144080030922  accuracy  0.9125
        Epoch  11  example  0  loss  4.442097568178135  accuracy  0.9125
        Epoch  11  example  800  loss  199.677455122896  accuracy  0.9
        Epoch  11  example  1600  loss  246.7080123814526  accuracy  0.8375
        Epoch  11  example  2400  loss  154.92769545231153  accuracy  0.875
        Epoch  11  example  3200  loss  171.8261726228939  accuracy  0.875
        Epoch  11  example  4000  loss  157.17389096830797  accuracy  0.9
        Epoch  11  example  4800  loss  182.41305225957717  accuracy  0.9125
        Epoch  12  example  0  loss  0.9192539582458412  accuracy  0.9
        Epoch  12  example  800  loss  174.45112140522264  accuracy  0.9125
        Epoch  12  example  1600  loss  236.26951824426385  accuracy  0.8875
        Epoch  12  example  2400  loss  168.44448881235124  accuracy  0.9375
        Epoch  12  example  3200  loss  161.3758694178804  accuracy  0.8875
        Epoch  12  example  4000  loss  121.63444886400845  accuracy  0.8875
        Epoch  12  example  4800  loss  168.23292962737827  accuracy  0.8875
        Epoch  13  example  0  loss  2.4610461313254293  accuracy  0.8875
        Epoch  13  example  800  loss  145.41376986905038  accuracy  0.925
        Epoch  13  example  1600  loss  212.45874246180222  accuracy  0.925
        Epoch  13  example  2400  loss  112.74015602177032  accuracy  0.9
        Epoch  13  example  3200  loss  131.17096781651802  accuracy  0.9
        Epoch  13  example  4000  loss  116.61585886798156  accuracy  0.925
        Epoch  13  example  4800  loss  153.7492115492334  accuracy  0.8875
        Epoch  14  example  0  loss  1.7785664799996281  accuracy  0.925
        Epoch  14  example  800  loss  133.25836918625672  accuracy  0.9125
        Epoch  14  example  1600  loss  209.49518821918846  accuracy  0.8625
        Epoch  14  example  2400  loss  114.41322823000631  accuracy  0.8875
        Epoch  14  example  3200  loss  156.0474600411849  accuracy  0.925
        Epoch  14  example  4000  loss  120.55243875403865  accuracy  0.9375
        Epoch  14  example  4800  loss  155.37190481019212  accuracy  0.875
        Epoch  15  example  0  loss  0.7626608242237936  accuracy  0.9125
        Epoch  15  example  800  loss  130.34396079721975  accuracy  0.9125
        Epoch  15  example  1600  loss  206.44221044293542  accuracy  0.8875
        Epoch  15  example  2400  loss  118.20356338223705  accuracy  0.9125
        Epoch  15  example  3200  loss  141.73814927181274  accuracy  0.875
        Epoch  15  example  4000  loss  115.59190227257888  accuracy  0.8875
        Epoch  15  example  4800  loss  168.8751375275004  accuracy  0.925
        Epoch  16  example  0  loss  0.4485878379322479  accuracy  0.9
        Epoch  16  example  800  loss  126.10549245964046  accuracy  0.925
        Epoch  16  example  1600  loss  215.8690985428633  accuracy  0.8375
        Epoch  16  example  2400  loss  125.72560975537554  accuracy  0.875
        Epoch  16  example  3200  loss  109.90217876548145  accuracy  0.9
        Epoch  16  example  4000  loss  91.85050603273002  accuracy  0.9125
        Epoch  16  example  4800  loss  128.03444068746703  accuracy  0.9
        Epoch  17  example  0  loss  0.4354281254573875  accuracy  0.925
        Epoch  17  example  800  loss  109.54722270222145  accuracy  0.925
        Epoch  17  example  1600  loss  189.90254114350964  accuracy  0.875
        Epoch  17  example  2400  loss  84.63382765412567  accuracy  0.9125
        Epoch  17  example  3200  loss  109.86055323132418  accuracy  0.9
        Epoch  17  example  4000  loss  104.02090171480994  accuracy  0.9
        Epoch  17  example  4800  loss  116.3379825516496  accuracy  0.9375
        Epoch  18  example  0  loss  0.1563607410722767  accuracy  0.9125
        Epoch  18  example  800  loss  126.16522812956302  accuracy  0.925
        Epoch  18  example  1600  loss  159.42238837033784  accuracy  0.8625
        Epoch  18  example  2400  loss  82.17130278461008  accuracy  0.8875
        Epoch  18  example  3200  loss  90.84354439302089  accuracy  0.9375
        Epoch  18  example  4000  loss  89.91880165404272  accuracy  0.9125
        Epoch  18  example  4800  loss  108.19408766853086  accuracy  0.9375
        Epoch  19  example  0  loss  0.4817844584263149  accuracy  0.9
        Epoch  19  example  800  loss  118.99357150940219  accuracy  0.9125
        Epoch  19  example  1600  loss  313.79675260970447  accuracy  0.775
        Epoch  19  example  2400  loss  127.15650072077614  accuracy  0.8875
        Epoch  19  example  3200  loss  126.99578496991556  accuracy  0.9375
        Epoch  19  example  4000  loss  129.483077631345  accuracy  0.8625
        Epoch  19  example  4800  loss  114.20412443435691  accuracy  0.9375
        Epoch  20  example  0  loss  0.6557615699278747  accuracy  0.9
        Epoch  20  example  800  loss  115.67342795165452  accuracy  0.9375
        Epoch  20  example  1600  loss  183.66254154897325  accuracy  0.875
        Epoch  20  example  2400  loss  92.8003875568032  accuracy  0.9125
        Epoch  20  example  3200  loss  81.5204536969444  accuracy  0.9375
        Epoch  20  example  4000  loss  92.34706839289464  accuracy  0.925
        Epoch  20  example  4800  loss  119.27729976185945  accuracy  0.925
        Epoch  21  example  0  loss  0.2695825293721237  accuracy  0.9
        Epoch  21  example  800  loss  92.71789158019223  accuracy  0.925
        Epoch  21  example  1600  loss  136.65207337450389  accuracy  0.9
        Epoch  21  example  2400  loss  81.4753003101732  accuracy  0.8625
        Epoch  21  example  3200  loss  94.78951186753046  accuracy  0.925
        Epoch  21  example  4000  loss  84.15092991476601  accuracy  0.95
        Epoch  21  example  4800  loss  104.87714693176308  accuracy  0.9375
        Epoch  22  example  0  loss  0.36698505611354904  accuracy  0.9125
        Epoch  22  example  800  loss  88.11166471587705  accuracy  0.925
        Epoch  22  example  1600  loss  121.48368354294408  accuracy  0.925
        Epoch  22  example  2400  loss  75.6750510076763  accuracy  0.95
        ```