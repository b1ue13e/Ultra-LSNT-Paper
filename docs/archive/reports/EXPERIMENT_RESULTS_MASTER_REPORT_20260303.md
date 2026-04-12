# Ultra-LSNT 实验结果总报告（中文版，规范证据）

- 生成时间：2026-03-03 01:43:38
- 规范证据目录：results/supplementary_evidence
- 规范证据文件数：153
- 覆盖范围：当前用于论文正文与答辩的核心/补充实验结果表
- 说明：表格字段名保留原始英文，数值与原始CSV完全一致。

## 一、主结果矩阵：Wind CN（干净 + 鲁棒）

### 1.1 主结论矩阵
数据来源：main4_matrix2_perf_robust_transfer_mainclaim.csv

| ﻿"category" | model | clean_R2 | clean_RMSE_kW | clean_MAE_kW | R2_sigma_eff_060 | R2_spike_06 | drop_pct_sigma060 | R2_wind_us | R2_air_quality | R2_gefcom_load | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| General DL | DLinear | 0.9379237294197084 | 6342.8623046875 | 4203.53466796875 | -1.579047679901123 | 0.20584636926651 | 268.3556554481218 | 0.8779157400131226 | 0.3432971835136413 | 0.7378214001655579 | 20260225T002754Z_verify_lite_fill |
| General DL | iTransformer | 0.9011505246162416 | 8004.04833984375 | 6333.2939453125 | 0.8717493414878845 | 0.8609154224395752 | 3.2626273108862724 | 0.8575311899185181 | 0.4672392010688782 | 0.7526589632034302 | 20260225T002754Z_verify_lite_fill |
| General DL | TimeMixer | 0.8885634541511536 | 8498.384765625 | 6661.359375 | 0.8336715698242188 | 0.7314769625663757 | 6.177598692634885 | 0.8390815854072571 | 0.3823473453521728 | 0.7281026840209961 | 20260225T002754Z_verify_lite_fill |
| General DL | Mamba (auditable) | 0.7129279375076294 | 17640.982285575825 | 13929.4755859375 | 0.6941477060317993 | 0.7112981081008911 | 2.6342398000923843 |  |  |  | 20260225T002754Z_verify_lite_fill |
| Hybrid Opt. | VMD-PSO-LSTM | 0.9105949401855468 | 9844.831740563168 | 7678.443359375 | 0.9057766199111938 | 0.9019150137901306 | 0.5291398031896845 |  |  |  | 20260225T002754Z_verify_lite_fill |
| Hybrid Opt. | CEEMDAN-WOA-GRU | 0.953388512134552 | 7108.4299250959775 | 4900.978515625 | 0.9384608268737792 | 0.9278420805931092 | 1.565750485848728 |  |  |  | 20260225T002754Z_verify_lite_fill |
| Traditional | ARIMA (auditable) | 0.9273094456570732 | 10575.104008359189 | 6075.502135187547 | 0.9090496222155018 | 0.9216091474050946 | 1.969118671989052 |  |  |  | 20260225T002754Z_verify_lite_fill |
| Traditional | SVR (auditable) | 0.9101129642642016 | 11759.6443201541 | 8266.440508269057 | 0.7268837572968943 | 0.8311609169097438 | 20.132578499773505 |  |  |  | 20260225T002754Z_verify_lite_fill |
| Traditional | SSA-ELM | 0.8894048929214478 | 10949.523825262904 | 8376.4296875 | 0.6957985758781433 | 0.8270354270935059 | 21.76807420154408 |  |  |  | 20260225T002754Z_verify_lite_fill |
| Edge-Native | Ultra-LSNT | 0.8727419376373291 | 9081.6591796875 | 7169.18603515625 | 0.7262898087501526 | 0.7829362154006958 | 16.783814110594506 | 0.7878 | 0.4861 | 0.7554 | 20260225T002754Z_verify_lite_fill |

## 二、跨域迁移结果

### 2.1 统一时间顺序协议下的迁移矩阵
数据来源：multi_domain_transfer_matrix_all_models.csv

| ﻿model | R2_wind_us | R2_air_quality | R2_gefcom_load | source |
| --- | --- | --- | --- | --- |
| Ultra-LSNT | 0.7878 | 0.4861 | 0.7554 | multi_domain_baselines + table_multi_domain |
| DLinear | 0.8779157102108002 | 0.3432970643043518 | 0.7370321154594421 | multi_domain_baselines + table_multi_domain |
| iTransformer | 0.8562867194414139 | 0.4589863419532776 | 0.7466928958892822 | multi_domain_baselines + table_multi_domain |
| TimeMixer | 0.8022965043783188 | 0.3868508338928222 | 0.7201057970523834 | multi_domain_baselines + table_multi_domain |
| ARIMA (auditable) | 0.6970844494597115 | 0.2008384571643823 | -0.403130070346585 | cross_domain_additional_models_clean |
| SVR (auditable) | 0.5795916275236808 | 0.17159943087470975 | 0.8226074235047375 | cross_domain_additional_models_clean |
| SSA-ELM | 0.8075498342514038 | 0.31515127420425415 | 0.7661759853363037 | cross_domain_additional_models_clean |
| Mamba (auditable) | 0.8865751624107361 | 0.3687828779220581 | -0.008335471153259277 | cross_domain_additional_models_clean |

## 三、鲁棒性压力测试（扩展）

### 3.1 两点高斯压力对比：Ultra-LSNT vs DLinear/LightGBM/Persistence
数据来源：robust_two_point_windcn.csv

| model | r2_sigma_0_0 | r2_sigma_0_2 | abs_drop_0_0_to_0_2 | rel_drop_pct_0_0_to_0_2 | r2_sigma_0_4 |
| --- | --- | --- | --- | --- | --- |
| DLinear | 0.93792375177145 | 0.2951500415802002 | 0.6427737101912498 | 68.5315526957546 | -1.579047679901123 |
| LightGBM | 0.9069388508796692 | 0.8221634328365326 | 0.0847754180431366 | 9.34742380491366 | 0.6155615746974945 |
| Persistence | 0.9174613952636719 | 0.903356671333313 | 0.014104723930358887 | 1.537364297089066 | 0.8611711263656616 |
| Ultra-LSNT | 0.8727746903896332 | 0.8552320450544357 | 0.0175426453351974 | 2.0099855699717737 | 0.7262898087501526 |

### 3.2 结构化SCADA故障（flatline/spike/dropout，强度0.2/0.6）
数据来源：structured_scada_fault_robustness_windcn.csv

| fault_type | severity | model | R2 | RMSE | MAE | R2_clean | R2_abs_drop_vs_clean | R2_rel_drop_pct_vs_clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.0 | Ultra-LSNT | 0.8727493286132812 | 9081.3955078125 | 7168.69189453125 | 0.8727493286132812 | 0.0 | 0.0 |
| clean | 0.0 | iTransformer | 0.9011505469679832 | 8004.0478515625 | 6333.2939453125 | 0.9011505469679832 | 0.0 | 0.0 |
| clean | 0.0 | TimeMixer | 0.8815930336713791 | 8760.142578125 | 6952.99169921875 | 0.8815930336713791 | 0.0 | 0.0 |
| clean | 0.0 | DLinear | 0.93792375177145 | 6342.8623046875 | 4203.53466796875 | 0.93792375177145 | 0.0 | 0.0 |
| flatline | 0.2 | Ultra-LSNT | 0.8721357583999634 | 9103.263671875 | 7181.29931640625 | 0.8727493286132812 | 0.0006135702133178711 | 0.07030314354899338 |
| flatline | 0.2 | iTransformer | 0.900520496070385 | 8029.515625 | 6351.89892578125 | 0.9011505469679832 | 0.0006300508975982666 | 0.06991627533469738 |
| flatline | 0.2 | TimeMixer | 0.8804119750857353 | 8803.7236328125 | 6982.3193359375 | 0.8815930336713791 | 0.0011810585856437683 | 0.13396868402252116 |
| flatline | 0.2 | DLinear | 0.9369067251682281 | 6394.6103515625 | 4230.7890625 | 0.93792375177145 | 0.0010170266032218933 | 0.10843382538303804 |
| flatline | 0.6 | Ultra-LSNT | 0.8664674758911133 | 9302.8515625 | 7345.27880859375 | 0.8727493286132812 | 0.006281852722167969 | 0.7197774339338946 |
| flatline | 0.6 | iTransformer | 0.8942484930157661 | 8278.7705078125 | 6541.4365234375 | 0.9011505469679832 | 0.006902053952217102 | 0.7659157479779373 |
| flatline | 0.6 | TimeMixer | 0.8702749609947205 | 9169.263671875 | 7253.43701171875 | 0.8815930336713791 | 0.01131807267665863 | 1.28382056622257 |
| flatline | 0.6 | DLinear | 0.9199313521385193 | 7203.669921875 | 4638.99072265625 | 0.93792375177145 | 0.017992399632930756 | 1.9183222089160912 |
| spike | 0.2 | Ultra-LSNT | 0.8508657515048981 | 9831.306640625 | 7737.6162109375 | 0.8727493286132812 | 0.02188357710838318 | 2.5074298416424083 |
| spike | 0.2 | iTransformer | 0.8897137567400932 | 8454.408203125 | 6594.5830078125 | 0.9011505469679832 | 0.011436790227890015 | 1.2691320297557729 |
| spike | 0.2 | TimeMixer | 0.7551197558641434 | 12597.9365234375 | 8853.1328125 | 0.8815930336713791 | 0.12647327780723572 | 14.345993329886005 |
| spike | 0.2 | DLinear | 0.7766384780406952 | 12031.6923828125 | 5335.3310546875 | 0.93792375177145 | 0.16128527373075485 | 17.19598991134796 |
| spike | 0.6 | Ultra-LSNT | 0.7829362899065018 | 11860.859375 | 9324.7802734375 | 0.8727493286132812 | 0.08981303870677948 | 10.29081727848295 |
| spike | 0.6 | iTransformer | 0.8609154522418976 | 9494.279296875 | 7228.6044921875 | 0.9011505469679832 | 0.04023509472608566 | 4.46485827051439 |
| spike | 0.6 | TimeMixer | 0.3428143858909607 | 20637.951171875 | 13905.2138671875 | 0.8815930336713791 | 0.5387786477804184 | 61.11421338445518 |
| spike | 0.6 | DLinear | 0.20584636926651 | 22686.8828125 | 7864.78369140625 | 0.93792375177145 | 0.73207738250494 | 78.0529740421085 |
| dropout | 0.2 | Ultra-LSNT | 0.8720718771219254 | 9105.537109375 | 7190.15185546875 | 0.8727493286132812 | 0.000677451491355896 | 0.077622688341944 |
| dropout | 0.2 | iTransformer | 0.9003921896219254 | 8034.6923828125 | 6363.0703125 | 0.9011505469679832 | 0.0007583573460578918 | 0.08415434564285243 |
| dropout | 0.2 | TimeMixer | 0.87986970692873 | 8823.6611328125 | 7002.68408203125 | 0.8815930336713791 | 0.0017233267426490784 | 0.19547871600939423 |
| dropout | 0.2 | DLinear | 0.9357994422316551 | 6450.47900390625 | 4275.57568359375 | 0.93792375177145 | 0.002124309539794922 | 0.22649064337935287 |
| dropout | 0.6 | Ultra-LSNT | 0.866717591881752 | 9294.134765625 | 7334.3896484375 | 0.8727493286132812 | 0.006031736731529236 | 0.691119034272202 |
| dropout | 0.6 | iTransformer | 0.894743837416172 | 8259.3583984375 | 6536.96630859375 | 0.9011505469679832 | 0.006406709551811218 | 0.7109477515568595 |
| dropout | 0.6 | TimeMixer | 0.8718839734792709 | 9112.2216796875 | 7211.9013671875 | 0.8815930336713791 | 0.009709060192108154 | 1.101308633494407 |
| dropout | 0.6 | DLinear | 0.9245377779006958 | 6993.3837890625 | 4557.9384765625 | 0.93792375177145 | 0.013385973870754242 | 1.427192119345762 |

### 3.3 深度SOTA模型高斯噪声扫描
数据来源：deep_sota_gaussian_windcn.csv

| sigma | model | R2 | RMSE | MAE |
| --- | --- | --- | --- | --- |
| 0.0 | PatchTST | 0.8666043281555176 | 9298.083984375 | 7342.15869140625 |
| 0.1 | PatchTST | 0.8658571243286133 | 9324.0869140625 | 7375.34375 |
| 0.2 | PatchTST | 0.8632093667984009 | 9415.658203125 | 7476.9423828125 |
| 0.3 | PatchTST | 0.8582322001457214 | 9585.4248046875 | 7646.86572265625 |
| 0.4 | PatchTST | 0.8507564067840576 | 9834.9091796875 | 7877.44189453125 |
| 0.0 | TimeMixer | 0.8885634541511536 | 8498.384765625 | 6661.359375 |
| 0.1 | TimeMixer | 0.8847201466560364 | 8643.69140625 | 6797.4814453125 |
| 0.2 | TimeMixer | 0.8737956285476685 | 9043.982421875 | 7132.13330078125 |
| 0.3 | TimeMixer | 0.8566054105758667 | 9640.263671875 | 7582.869140625 |
| 0.4 | TimeMixer | 0.8336715698242188 | 10382.59375 | 8105.57470703125 |
| 0.0 | iTransformer | 0.9011505246162415 | 8004.04833984375 | 6333.2939453125 |
| 0.1 | iTransformer | 0.8993415832519531 | 8076.953125 | 6408.2578125 |
| 0.2 | iTransformer | 0.8937150239944458 | 8299.625 | 6613.70849609375 |
| 0.3 | iTransformer | 0.884404718875885 | 8655.5087890625 | 6914.81884765625 |
| 0.4 | iTransformer | 0.8717493414878845 | 9117.0078125 | 7285.0966796875 |

### 3.4 缺失 + 插补敏感性（ffill/linear，p=0.1/0.3）
数据来源：missingness_imputation_ultra_dlinear_summary.csv

| model | imputation | missing_rate_p | R2_mean | R2_std | RMSE_mean | RMSE_std | MAE_mean | MAE_std | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear | ffill | 0.1 | 0.9325670202573141 | 9.070860233297248e-05 | 6610.870768229167 | 4.445287649647318 | 4409.85693359375 | 2.094871163879193 | 3 |
| Ultra-LSNT | ffill | 0.1 | 0.8718065619468689 | 0.00010123460696657002 | 9114.974283854166 | 3.597643570216902 | 7195.704264322917 | 1.7295425266985207 | 3 |
| DLinear | ffill | 0.3 | 0.9241774479548136 | 0.0005172423430113477 | 7010.032552083333 | 23.933522032504833 | 4725.010579427083 | 12.48176975622992 | 3 |
| Ultra-LSNT | ffill | 0.3 | 0.8692392706871033 | 0.00023839604748116804 | 9205.791015625 | 8.389676173602448 | 7267.152018229167 | 7.682115581671065 | 3 |
| DLinear | linear | 0.1 | 0.9325111508369446 | 0.00012233854429149696 | 6613.609049479167 | 5.994555413711033 | 4411.297526041667 | 2.659285407942194 | 3 |
| Ultra-LSNT | linear | 0.1 | 0.8721594015757242 | 8.038552819158186e-05 | 9102.421223958334 | 2.861782096114916 | 7184.6279296875 | 0.8092216965112924 | 3 |
| DLinear | linear | 0.3 | 0.9239373008410136 | 0.0006037622602843492 | 7021.116373697917 | 27.89739072287434 | 4737.261881510417 | 16.5164536479078 | 3 |
| Ultra-LSNT | linear | 0.3 | 0.8704546093940735 | 0.00019468133458645445 | 9162.910807291666 | 6.882632463563847 | 7229.373046875 | 4.61381857098369 | 3 |

### 3.5 概率预测指标（NLL/CRPS/PICP/PINAW）
数据来源：probabilistic_gaussian_minimal_ultra_dlinear.csv

| model | calibrated_sigma_kw | gaussian_nll | gaussian_crps | PICP_80 | PINAW_80 | PICP_90 | PINAW_90 | n_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ultra-LSNT | 5044.081480113338 | 10.131403294545894 | 3353.466664160422 | 0.7253050632212468 | 0.10994651740509359 | 0.824195240461598 | 0.14111482735975595 | 18111 |
| DLinear | 1329.7430340666156 | 8.611825753260298 | 670.2736876271574 | 0.8725636353597261 | 0.02898458643376671 | 0.9268952570261167 | 0.037201313940894626 | 18111 |

## 四、硬件与效率结果

### 4.1 硬件与安全联合矩阵
数据来源：main4_matrix3_hardware_safety_joint.csv

| panel | item | latency_ms | throughput_hz | p95_ms | model_size_mib | clean_R2 | R2_sigma_eff_060 | safety_or_control | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_hardware | Ultra-LSNT(full) | 3.713820800010581 | 269.26447285694314 | 4.331135004031238 |  | 0.8727419376373291 | 0.7262898087501526 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | Ultra-LSNT-Lite(opt) | 1.1710027990629897 | 853.9689237294546 | 1.6300449933623893 |  | 0.8636489808559418 | 0.8281462341547012 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | Mamba (auditable) | 0.8612349985115644 | 1161.1232726587484 | 1.1093149922089645 | 0.1468219757080078 | 0.7129279375076294 | 0.6941477060317993 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | DLinear | 0.2654196321964264 | 3767.6188144965113 |  |  | 0.9379237294197084 | -1.579047679901123 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | VMD-PSO-LSTM | 10.296521669079084 | 97.1201763215868 | 11.974505026591942 | 0.246826171875 | 0.9105949401855468 | 0.8898751735687256 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | CEEMDAN-WOA-GRU | 11.081744165858254 | 90.23850262496585 | 11.6581549460534 | 0.21295166015625 | 0.953388512134552 | 0.9262365102767944 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | ARIMA (auditable) | 3.055398892611265 | 327.2895078996904 | 3.1065803952515108 | 3.814697265625e-05 | 0.9301621585489368 | 0.8951691996469062 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | SVR (auditable) | 3.8120673317462206 | 262.32485236348725 | 3.873562626540661 | 0.540985107421875 | 0.9101129642642012 | 0.7268837572968943 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| A_hardware | SSA-ELM | 0.0365548936480825 | 27356.11843456848 | 0.0435808673501014 | 0.8107147216796875 | 0.8894048929214478 | 0.6957985758781433 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| B_safety_ablation | Ultra-LSNT (no guardrail) |  |  |  |  | 0.8727493286132812 | 0.8487721681594849 | Over-rated 0.0653% | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| B_safety_ablation | FG-MoE (full-trained) |  |  |  |  | 0.8746772408485413 | 0.8525823950767517 | Over-rated 0.0023% | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| B_safety_ablation | Jump Gate tau=0.3 | 7.026500568132509 | 142.31835467790737 |  |  | -inf |  | Skip 33.38% | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| B_safety_ablation | Jump Gate tau=0.5 | 6.978213829411701 | 143.30314668564765 |  |  | -inf |  | Skip 0.00% | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| C_joint_param | K=1, tau=0.7, lambda2=0.01 | 10.60885332893425 | 94.26089408481421 |  |  |  | 0.9301179093360452 | Over-rated 19.1332% | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| C_joint_param | K=2, tau=0.5, lambda2=0.01 | 14.353413329808973 | 69.66983929343229 |  |  |  | 0.929446572170238 | Over-rated 19.0091% | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| C_joint_param | K=4, tau=0.9, lambda2=0.01 | 19.76897999411449 | 50.58429925558701 |  |  |  | 0.9298515859168932 | Over-rated 18.9690% | 20260226T215438Z_main4_matrix_build_submission_oneclick |
| C_joint_param | K=1, tau=0.9, lambda2=1.00 | 10.231653325414904 | 97.73591502714925 |  |  |  | 0.3830673004070142 | Over-rated 0.0000% | 20260226T215438Z_main4_matrix_build_submission_oneclick |

### 4.2 ARM时延回填表
数据来源：latency_arm_full_backfill.csv

| model | mean_ms | p95_ms | std_ms | memory_mb | params |
| --- | --- | --- | --- | --- | --- |
| Ultra-LSNT(full) | 8.79495082423091 | 10.418968088924878 | 1.1032086938513923 | 0.14453125 | 267672.0 |
| Ultra-LSNT-Lite(orig) | 4.632404167205095 | 5.389842018485067 | 0.4797074292165594 | 0.2109375 | 40344.0 |
| Ultra-LSNT-Lite(opt) | 5.00746488571167 | 5.900161154568195 | 0.6171250140407184 | 0.0859375 | 18712.0 |
| DLinear | 0.1019650138914585 | 0.1048268750309944 | 0.0034280943422311 | 0.0 | 2328.0 |
| TimeMixer | 1.8110505305230613 | 2.1201904863119125 | 0.2353405751742986 | 0.15625 | 4248.0 |
| iTransformer | 1.3821213878691196 | 1.942608878016472 | 0.3705919474270164 | 0.1015625 | 4248.0 |
| Mamba (auditable) | 0.8612349985115644 | 1.1093149922089645 | 0.1366617038142709 | 0.1468219757080078 | 35832.0 |
| VMD-PSO-LSTM | 10.296521669079084 | 11.974505026591942 | 0.791077788829818 | 0.246826171875 | 64704.0 |
| CEEMDAN-WOA-GRU | 11.081744165858254 | 11.6581549460534 | 0.3715800557198354 | 0.21295166015625 | 55824.0 |
| ARIMA (auditable) | 2.1831799994106404 | 2.670254962868057 | 0.2362604011067206 | 3.0517578125e-05 | 4.0 |
| SVR (auditable) | 2.750382501690183 | 3.4401899989461526 | 0.3190621792137748 | 0.540985107421875 | 70908.0 |
| SSA-ELM | 0.0235795461445708 | 0.0285149842966347 | 0.0103443495806295 | 0.8107147216796875 | 106262.0 |

### 4.3 CPU batch-1时延：DLinear vs LightGBM
数据来源：cpu_batch1_latency_table.csv

| model | device | batch1_latency_ms | batch1_latency_std_ms | batch1_freq_hz | batch4_latency_ms_reported | estimated_energy_mj_batch1 | power_assumption_w | source_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear | cpu | 0.2654196321964264 | 0.0843380371815598 | 3767.6188144965113 | 0.4595883190631866 | 0.7962588965892792 | 3.0 | dlinear_latency_results.csv |
| LightGBM | cpu | 50.39703272283077 | 21.134684403987546 | 19.842438055821923 | 50.59029050171375 | 151.19109816849232 | 3.0 | lightgbm_latency_results.csv |

### 4.4 Ultra系列 CPU batch-1时延
数据来源：cpu_batch1_ultra_family_latency.csv

| model | threads | warmup_runs | measure_runs | latency_mean_ms | latency_std_ms | latency_p50_ms | latency_p95_ms | latency_min_ms | latency_max_ms | peak_rss_mb | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ultra-LSNT-Lite(opt) | 1 | 50 | 250 | 1.4669886454939842 | 0.014726951931224728 | 1.4638686552643776 | 1.498419139534235 | 1.4400836080312729 | 1.5274230390787125 | 1080.12109375 | arch-only (optimized config hidden=96,blocks=1,experts=2,top_k=1) |
| Ultra-LSNT-Lite(orig) | 1 | 50 | 250 | 3.2606254294514656 | 0.02670540222029859 | 3.2577533274888992 | 3.3108647912740707 | 3.203839063644409 | 3.369007259607315 | 1080.12109375 | arch-only (no dedicated lite checkpoint found) |
| Ultra-LSNT(full) | 1 | 50 | 250 | 6.972947709262371 | 0.08917461646293748 | 6.9410791620612144 | 7.142783235758543 | 6.838725879788399 | 7.268842309713364 | 1080.12109375 | checkpoint: checkpoints_ts/main/best_model.pth |

## 五、护栏机制与物理安全闭环

### 5.1 三组对照：无护栏 vs 仅裁剪 vs FG-MoE
数据来源：fgmoe_clip_only_control_windcn.csv

| scenario | sigma_cfg | sigma_eff | model | R2 | RMSE | MAE | neg_rate_pct | over_rated_rate_pct | cut_region_nonzero_rate_pct | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.0 | 0.0 | Ultra-LSNT (no guardrail) | 0.8727493286132812 | 9081.3955078125 | 7168.69189453125 | 0.0 | 0.11641175712734433 | 0.9386560653746342 | 20260301T153351Z_clip_only_guardrail |
| clean | 0.0 | 0.0 | Ultra-LSNT + clip-only (posthoc) | 0.43185412883758545 | 19189.005859375 | 14757.9912109375 | 0.0 | 0.0 | 0.0 | 20260301T153351Z_clip_only_guardrail |
| gaussian_60pct | 0.4 | 0.6 | Ultra-LSNT (no guardrail) | 0.8502360582351685 | 9852.041015625 | 7849.66845703125 | 0.0 | 0.06556788691955165 | 1.6895809176743415 | 20260301T153351Z_clip_only_guardrail |
| gaussian_60pct | 0.4 | 0.6 | Ultra-LSNT + clip-only (posthoc) | 0.3773219585418701 | 20088.81640625 | 15335.8349609375 | 0.0 | 0.0 | 0.0 | 20260301T153351Z_clip_only_guardrail |
| clean | 0.0 | 0.0 | FG-MoE (full-trained) | 0.8746772408485413 | 9012.33984375 | 6986.36767578125 | 0.0 | 0.0 | 0.9386560653746342 | 20260301T153351Z_clip_only_guardrail |
| gaussian_60pct | 0.4 | 0.6 | FG-MoE (full-trained) | 0.8525823950767517 | 9774.560546875 | 7691.5302734375 | 0.0 | 0.0023006276112123 | 1.7558389928772569 | 20260301T153351Z_clip_only_guardrail |

### 5.2 物理闭环：高估/低估、爬坡、包络偏差
数据来源：physics_closure.csv

| model | mode | R2 | RMSE | MAE | neg_rate_pct | over_rated_rate_pct | cut_region_nonzero_rate_pct | under_rated_low_output_rate_pct | under_rated_denom_points | ramp_rate_violation_pct | ramp_rate_violation_limit_kw | envelope_deviation_area_mean_kw | envelope_deviation_area_p95_kw | envelope_deviation_area_sum_kw | envelope_deviation_area_pct_of_rated | run_id | scenario | fault | severity | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FG-MoE (full-trained) | full_guardrail | 0.8746772110462189 | 9012.3408203125 | 6986.36669921875 | 0.0 | 0.0 | 0.9386560653746342 | 8.725071225071225 | 28080 | 2.467393104839 | 3198.0 | 15519.74609375 | 38543.615234374985 | 6745874944.0 | 12.688549944609322 | 20260302T145602Z_guardrail_physics_closure | clean | clean | 0.0 | 42 |
| Ultra-LSNT (no guardrail) | no_guardrail | 0.8727493286132812 | 9081.3955078125 | 7168.69189453125 | 0.0 | 0.11641175712734433 | 0.9386560653746342 | 7.482193732193732 | 28080 | 2.5422935376770783 | 3198.0 | 16070.4248046875 | 39458.441796875 | 6985234944.0 | 13.138770862203936 | 20260302T145602Z_guardrail_physics_closure | clean | clean | 0.0 | 42 |
| Ultra-LSNT + clip-only (posthoc) | clip_only | 0.43185436725616455 | 19189.001953125 | 14757.9892578125 | 0.0 | 0.0 | 0.0 | 7.482193732193732 | 28080 | 1.1633573638888688 | 3198.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20260302T145602Z_guardrail_physics_closure | clean | clean | 0.0 | 42 |
| FG-MoE (full-trained) | full_guardrail | 0.7863720953464508 | 11766.615234375 | 9584.544921875 | 0.0 | 0.11503138056061693 | 0.011043012533819226 | 17.653425006366184 | 62832 | 2.9134347850093505 | 3198.0 | 10654.275390625 | 31710.86171874999 | 4631029760.0 | 8.710664762228872 | 20260302T145602Z_guardrail_physics_closure | drift_60pct | drift | 0.6 | 42 |
| Ultra-LSNT (no guardrail) | no_guardrail | 0.7504201531410217 | 12718.2490234375 | 10663.3564453125 | 0.0 | 0.2208602506763845 | 0.011043012533819226 | 16.56003310415075 | 62832 | 3.1635830254493427 | 3198.0 | 11565.7783203125 | 33961.610546874996 | 5027227648.0 | 9.455886390091404 | 20260302T145602Z_guardrail_physics_closure | drift_60pct | drift | 0.6 | 42 |
| Ultra-LSNT + clip-only (posthoc) | clip_only | 0.6712816059589386 | 14596.0234375 | 11402.12109375 | 0.0 | 0.0 | 0.0 | 16.56003310415075 | 62832 | 1.9985452031314141 | 3198.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20260302T145602Z_guardrail_physics_closure | drift_60pct | drift | 0.6 | 42 |
| FG-MoE (full-trained) | full_guardrail | 0.8286466002464294 | 10538.259765625 | 8313.6796875 | 0.0 | 0.009202510444849354 | 2.7055380707857104 | 24.577336074937172 | 35016 | 2.2472530506322124 | 3198.0 | 16264.134765625 | 42482.10917968745 | 7069433856.0 | 13.297143202787112 | 20260302T145602Z_guardrail_physics_closure | gaussian_60pct | gaussian | 0.6 | 42 |
| Ultra-LSNT (no guardrail) | no_guardrail | 0.8273229598999023 | 10578.8837890625 | 8447.4443359375 | 0.0 | 0.054984999907974896 | 2.7055380707857104 | 22.906671236006396 | 35016 | 2.543733930616272 | 3198.0 | 16932.73046875 | 43512.70273437497 | 7360048640.0 | 13.843770056126495 | 20260302T145602Z_guardrail_physics_closure | gaussian_60pct | gaussian | 0.6 | 42 |
| Ultra-LSNT + clip-only (posthoc) | clip_only | 0.31197088956832886 | 21116.6953125 | 16071.5732421875 | 0.0 | 0.0 | 0.0 | 22.906671236006396 | 35016 | 1.098059550645416 | 3198.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20260302T145602Z_guardrail_physics_closure | gaussian_60pct | gaussian | 0.6 | 42 |
| FG-MoE (full-trained) | full_guardrail | 0.872055396437645 | 9106.123046875 | 7069.443359375 | 0.0 | 0.0004601255222424677 | 0.9441775716415438 | 9.011130136986301 | 28032 | 2.4361845911564677 | 3198.0 | 15492.0537109375 | 38679.341015624996 | 6733837824.0 | 12.665909356272433 | 20260302T145602Z_guardrail_physics_closure | missingness_60pct | missingness | 0.6 | 42 |
| Ultra-LSNT (no guardrail) | no_guardrail | 0.8707137256860733 | 9153.744140625 | 7222.79736328125 | 0.0 | 0.09271529273185725 | 0.9441775716415438 | 7.791095890410959 | 28032 | 2.5254889533864837 | 3198.0 | 16046.8828125 | 39476.35742187496 | 6975002112.0 | 13.11952352775257 | 20260302T145602Z_guardrail_physics_closure | missingness_60pct | missingness | 0.6 | 42 |
| Ultra-LSNT + clip-only (posthoc) | clip_only | 0.4286125898361206 | 19243.669921875 | 14798.12890625 | 0.0 | 0.0 | 0.0 | 7.791095890410959 | 28032 | 1.1671984117267191 | 3198.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20260302T145602Z_guardrail_physics_closure | missingness_60pct | missingness | 0.6 | 42 |
| FG-MoE (full-trained) | full_guardrail | 0.8740952163934708 | 9033.2421875 | 7005.93798828125 | 0.0 | 0.0 | 0.8668764839048092 | 9.765625 | 28416 | 2.442426293892974 | 3198.0 | 15521.1376953125 | 38786.0357421875 | 6746479616.0 | 12.689687682676821 | 20260302T145602Z_guardrail_physics_closure | quantization_60pct | quantization | 0.6 | 42 |
| Ultra-LSNT (no guardrail) | no_guardrail | 0.8720932751893997 | 9104.775390625 | 7186.255859375 | 0.0 | 0.12377376548322383 | 0.8668764839048092 | 8.410754504504505 | 28416 | 2.531970721612856 | 3198.0 | 16076.5947265625 | 39707.6337890625 | 6987916800.0 | 13.143815233509521 | 20260302T145602Z_guardrail_physics_closure | quantization_60pct | quantization | 0.6 | 42 |
| Ultra-LSNT + clip-only (posthoc) | clip_only | 0.4308353662490845 | 19206.203125 | 14766.7177734375 | 0.0 | 0.0 | 0.0 | 8.410754504504505 | 28416 | 1.1770410968112102 | 3198.0 | 0.0 | 0.0 | 0.0 | 0.0 | 20260302T145602Z_guardrail_physics_closure | quantization_60pct | quantization | 0.6 | 42 |

## 六、调度耦合结果

### 6.1 Raw vs Mapped 调度汇总
数据来源：dispatch_raw_vs_mapped_summary.csv

| model | raw_opt_ratio | raw_fallback_pct | raw_cost_mean_cny | raw_curtail_mean_kwh | raw_nonfinite_days | mapped_opt_ratio | mapped_fallback_pct | mapped_cost_mean_cny | mapped_curtail_mean_kwh | mapped_nonfinite_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear | 0.0 | 100.0 | 694873.6117890625 | 2461226.863151042 | 0 | 0.375 | 62.5 | 453913.2984250967 | 464231.8920632628 | 0 |
| TimeMixer | 0.0 | 100.0 | 704920.4465703125 | 2628674.109505208 | 0 | 0.7833333333333333 | 21.666666666666668 | 290831.1176100786 | 163330.32675679625 | 0 |
| iTransformer | 0.0 | 100.0 | 708595.5473632812 | 2689925.789388021 | 0 | 0.8333333333333334 | 16.666666666666664 | 268012.5018993344 | 126838.99707350506 | 0 |
| Ultra-LSNT | 0.0 | 100.0 |  |  | 120 | 0.7583333333333333 | 24.166666666666668 | 299763.1789363524 | 184074.7266336355 | 0 |

### 6.2 主调度价值表（L1 + L2）
数据来源：main4_matrix4_dispatch_value.csv

| layer | model | operating_cost_million | curtailment_million | fallback_pct | improvement_vs_dlinear_pct | run_id | curtailment_mwh | opf_success_pct | curtailment_improve_vs_dlinear_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1_CopperPlate_120d | iTransformer | 32.16150022792013 | 15.220679648820607 | 16.666666666666664 | 40.955133319681536 | 20260226T215438Z_main4_matrix_build_submission_oneclick |  |  |  |
| L1_CopperPlate_120d | TimeMixer | 34.89973411320943 | 19.59963921081555 | 21.666666666666668 | 35.928046475141855 | 20260226T215438Z_main4_matrix_build_submission_oneclick |  |  |  |
| L1_CopperPlate_120d | Ultra-LSNT | 35.97158147236229 | 22.08896719603626 | 24.16666666666667 | 33.96025629202438 | 20260226T215438Z_main4_matrix_build_submission_oneclick |  |  |  |
| L1_CopperPlate_120d | DLinear | 54.46959581101161 | 55.707827047591536 | 62.5 |  | 20260226T215438Z_main4_matrix_build_submission_oneclick |  |  |  |
| L2_RTS24_DCOPF_wmax260 | iTransformer | 145.43447578861083 |  |  |  | 20260226T215438Z_main4_matrix_build_submission_oneclick | 54242.21212749597 | 100.0 | 12.231350902489478 |
| L2_RTS24_DCOPF_wmax260 | TimeMixer | 147.91294868785633 |  |  |  | 20260226T215438Z_main4_matrix_build_submission_oneclick | 54242.21212791165 | 100.0 | 12.23135090181688 |
| L2_RTS24_DCOPF_wmax260 | Ultra-LSNT | 145.5912513359587 |  |  |  | 20260226T215438Z_main4_matrix_build_submission_oneclick | 54381.53079215698 | 100.0 | 12.005921103228635 |
| L2_RTS24_DCOPF_wmax260 | DLinear | 137.07858785975966 |  |  |  | 20260226T215438Z_main4_matrix_build_submission_oneclick | 61801.35240230615 | 99.96527777777776 |  |

### 6.3 RTS-24 DC-OPF汇总（wmax260）
数据来源：dispatch_network_ieee24_dcopf_wmax260_aggregate.csv

| model | days | total_rt_cost | mean_rt_cost_per_day | total_curtailment_mwh | mean_curtailment_mwh_per_day | total_slack_mwh | mean_slack_mwh_per_day | total_congestion_hours | total_opf_success_hours | congestion_hour_ratio | opf_success_ratio | cost_reduction_vs_dlinear_pct | curtailment_reduction_vs_dlinear_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear | 120 | 137078587.85975966 | 1142321.5654979972 | 61801.35240230615 | 515.0112700192179 | 4395.407690708197 | 36.62839742256831 | 1933 | 2879 | 0.6711805555555556 | 0.9996527777777777 | 0.0 | 0.0 |
| iTransformer | 120 | 145434475.78861082 | 1211953.9649050902 | 54242.21212749597 | 452.0184343957998 | 8630.438727751956 | 71.9203227312663 | 2235 | 2880 | 0.7760416666666666 | 1.0 | -6.095691573216217 | 12.231350902489478 |
| Ultra-LSNT | 120 | 145591251.3359587 | 1213260.4277996558 | 54381.53079215698 | 453.1794232679748 | 8708.405749083024 | 72.57004790902519 | 2221 | 2880 | 0.7711805555555555 | 1.0 | -6.210060673303726 | 12.005921103228635 |
| TimeMixer | 120 | 147912948.68785632 | 1232607.905732136 | 54242.21212791165 | 452.01843439926375 | 9878.612670581533 | 82.32177225484611 | 2235 | 2880 | 0.7760416666666666 | 1.0 | -7.903758710427419 | 12.23135090181688 |

### 6.4 映射敏感性（wmax260/280/300）
数据来源：dispatch_mapping_sensitivity_wmax260_280_300.csv

| mapping_case | wind_max_mw | model | total_rt_cost | total_curtailment_mwh | opf_success_ratio | cost_reduction_vs_dlinear_pct | curtailment_reduction_vs_dlinear_pct | source_file | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wmax260 | 260 | DLinear | 137078587.85975966 | 61801.35240230615 | 0.9996527777777776 | 0.0 | 0.0 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax260 | 260 | iTransformer | 145434475.78861082 | 54242.21212749597 | 1.0 | -6.095691573216217 | 12.231350902489478 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax260 | 260 | TimeMixer | 147912948.68785632 | 54242.21212791165 | 1.0 | -7.903758710427419 | 12.23135090181688 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax260 | 260 | Ultra-LSNT | 145591251.3359587 | 54381.53079215698 | 1.0 | -6.210060673303726 | 12.005921103228635 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax280 | 280 | DLinear | 133139157.0485513 | 97778.61552259812 | 1.0 | 0.0 | 0.0 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax280 | 280 | iTransformer | 134498058.46768847 | 93914.96479411409 | 1.0 | -1.0206624777124134 | 3.951427117099122 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax280 | 280 | TimeMixer | 134688881.9854964 | 93914.96479299047 | 1.0 | -1.163988845430322 | 3.95142711824826 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax280 | 280 | Ultra-LSNT | 134612548.31785616 | 93930.39764416282 | 1.0 | -1.1066550982950576 | 3.935643655688621 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax300 | 300 | DLinear | 130069593.16153587 | 139403.69283563524 | 1.0 | 0.0 | 0.0 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax300 | 300 | iTransformer | 130060475.2900599 | 138093.39426021092 | 1.0 | 0.0070099946147067 | 0.9399310368120898 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax300 | 300 | TimeMixer | 130060475.29013842 | 138093.39426051825 | 1.0 | 0.007009994554332 | 0.9399310365916248 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |
| wmax300 | 300 | Ultra-LSNT | 130060475.29005568 | 138093.3942602722 | 1.0 | 0.0070099946179488 | 0.939931036768122 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_aggregate.csv | 20260301T165639Z_dispatch_mapping_sensitivity_archival |

### 6.5 日级一致性（相对DLinear的胜率）
数据来源：dispatch_mapping_daylevel_consistency_wmax260_280_300.csv

| mapping_case | wind_max_mw | model | metric | n_days | day_win_ratio_vs_dlinear | day_win_ci95_low | day_win_ci95_high | delta_mean | delta_median | delta_p25 | delta_p75 | source_daily_file | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wmax260 | 260 | iTransformer | rt_cost_total | 120 | 0.4166666666666667 | 0.3333333333333333 | 0.5083333333333333 | 69632.39940709298 | 1195.4710613653297 | -27.482656422595028 | 118605.95823535055 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax260 | 260 | TimeMixer | rt_cost_total | 120 | 0.35833333333333334 | 0.275 | 0.44166666666666665 | 90286.34023413881 | 1195.4710661212448 | -27.482655033585615 | 142807.0348224783 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax260 | 260 | Ultra-LSNT | rt_cost_total | 120 | 0.425 | 0.3333333333333333 | 0.5166666666666667 | 70938.86230165859 | 1195.4710614787182 | -27.48265723598888 | 114089.53140877269 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax260 | 260 | iTransformer | curtailment_mwh_total | 120 | 0.8333333333333334 | 0.7666666666666667 | 0.9 | -62.99283562341816 | -0.24178162207920195 | -49.33190796889659 | -2.330860837673754e-09 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax260 | 260 | TimeMixer | curtailment_mwh_total | 120 | 0.7166666666666667 | 0.6333333333333333 | 0.7916666666666666 | -62.992835619954185 | -0.2417816227067533 | -49.331907958774025 | 3.0529569983173133e-09 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax260 | 260 | Ultra-LSNT | curtailment_mwh_total | 120 | 0.8583333333333333 | 0.7916666666666666 | 0.9166666666666666 | -61.831846751243134 | -0.2417816332021232 | -49.33190795756718 | -3.867256737066782e-09 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax260_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax280 | 280 | iTransformer | rt_cost_total | 120 | 0.30833333333333335 | 0.225 | 0.4 | 11324.178492809811 | 2.0551960915327072e-06 | -5.884794518351555e-07 | 4.794099368155003e-06 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax280 | 280 | TimeMixer | rt_cost_total | 120 | 0.49166666666666664 | 0.4 | 0.5752083333333321 | 12914.374474542386 | 7.834751158952713e-08 | -2.0577572286128998e-06 | 2.530112396925688e-06 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax280 | 280 | Ultra-LSNT | rt_cost_total | 120 | 0.35 | 0.26666666666666666 | 0.44166666666666665 | 12278.260577540424 | 1.1255033314228058e-06 | -1.4575198292732239e-06 | 3.893161192536354e-06 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax280 | 280 | iTransformer | curtailment_mwh_total | 120 | 0.48333333333333334 | 0.39166666666666666 | 0.5666666666666667 | -32.19708940403357 | 1.3835403933626367e-09 | -9.48520550991816e-09 | 1.3570570445153862e-08 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax280 | 280 | TimeMixer | curtailment_mwh_total | 120 | 0.7166666666666667 | 0.6333333333333333 | 0.8 | -32.19708941339703 | -6.6833081291406415e-09 | -1.6677972780598793e-08 | 7.944294111439376e-10 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax280 | 280 | Ultra-LSNT | curtailment_mwh_total | 120 | 0.48333333333333334 | 0.39166666666666666 | 0.5666666666666667 | -32.06848232029422 | 1.3252758890303085e-09 | -1.234707269759383e-08 | 9.508823950454826e-09 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax280_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax300 | 300 | iTransformer | rt_cost_total | 120 | 0.6166666666666667 | 0.5333333333333333 | 0.7002083333333321 | -75.98226229996266 | -7.710186764597893e-07 | -2.9099173843860626e-06 | 6.585032679140568e-07 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax300 | 300 | TimeMixer | rt_cost_total | 120 | 0.5416666666666666 | 0.4583333333333333 | 0.6333333333333333 | -75.98226164557467 | -1.3562384992837906e-07 | -2.3765023797750473e-06 | 1.5025725588202477e-06 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax300 | 300 | Ultra-LSNT | rt_cost_total | 120 | 0.5833333333333334 | 0.49166666666666664 | 0.6666666666666666 | -75.98226233514919 | -4.0314625948667526e-07 | -2.9990915209054947e-06 | 9.878422133624554e-07 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax300 | 300 | iTransformer | curtailment_mwh_total | 120 | 0.575 | 0.48333333333333334 | 0.6666666666666666 | -10.919154795202598 | -2.1985329112794716e-09 | -1.0151836704608286e-08 | 6.1264842088348814e-09 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax300 | 300 | TimeMixer | curtailment_mwh_total | 120 | 0.5 | 0.4083333333333333 | 0.5833333333333334 | -10.91915479264138 | 6.342588676488958e-10 | -9.791619959287345e-09 | 9.920768206939101e-09 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |
| wmax300 | 300 | Ultra-LSNT | curtailment_mwh_total | 120 | 0.5166666666666667 | 0.425 | 0.6083333333333333 | -10.919154794691773 | -3.1013769330456853e-10 | -9.657412647356978e-09 | 6.858044798718765e-09 | results/supplementary_evidence/dispatch_network_ieee24_dcopf_wmax300_daily.csv | 20260302T014341Z_dispatch_mapping_daylevel_consistency_archival |

## 七、联合参数与帕累托分析

### 7.1 (K, tau, lambda2)帕累托前沿
数据来源：joint_sensitivity_pareto_frontier.csv

| K | tau | lambda2 | sigma_eff | R2 | latency_ms | over_rated_pct | negative_pct | cut_region_nonzero_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.9 | 1.0 | 0.6 | 0.3830673004070142 | 10.231653325414905 | 0.0 | 0.0 | 0.0 |
| 1 | 0.9 | 0.1 | 0.6 | 0.9257051812568343 | 10.231653325414905 | 18.01689995659722 | 0.0 | 1.4485677083333335 |
| 1 | 0.9 | 0.01 | 0.6 | 0.9290630233442296 | 10.231653325414905 | 18.690321180555554 | 0.0 | 1.4485677083333335 |
| 1 | 0.7 | 1.0 | 0.6 | 0.3889676968038195 | 10.60885332893425 | 0.0 | 0.0 | 0.0 |
| 1 | 0.7 | 0.1 | 0.6 | 0.9266408372510521 | 10.60885332893425 | 18.33224826388889 | 0.0 | 1.611328125 |
| 1 | 0.7 | 0.01 | 0.6 | 0.9301179093360452 | 10.60885332893425 | 19.133165147569446 | 0.0 | 1.611328125 |
| 1 | 0.5 | 0.01 | 0.6 | 0.9292664872882449 | 10.98017499219471 | 18.990071614583336 | 0.0 | 1.3834635416666665 |
| 2 | 0.3 | 1.0 | 0.6 | 0.38961985918960373 | 13.27047333373533 | 0.0 | 0.0 | 0.0 |
| 2 | 0.3 | 0.1 | 0.6 | 0.9258487172357927 | 13.27047333373533 | 17.972819010416664 | 0.0 | 1.6927083333333333 |
| 2 | 0.3 | 0.01 | 0.6 | 0.9290894726242014 | 13.27047333373533 | 18.73101128472222 | 0.0 | 1.6927083333333333 |
| 2 | 0.5 | 0.01 | 0.6 | 0.929446572170238 | 14.353413329808973 | 19.00906032986111 | 0.0 | 1.4973958333333335 |
| 4 | 0.5 | 1.0 | 0.6 | 0.39131822845228614 | 18.18168999743648 | 0.0 | 0.0 | 0.0 |
| 4 | 0.9 | 0.1 | 0.6 | 0.9262758762907225 | 19.76897999411449 | 18.165418836805554 | 0.0 | 1.7740885416666667 |
| 4 | 0.9 | 0.01 | 0.6 | 0.9298515859168933 | 19.76897999411449 | 18.96904839409722 | 0.0 | 1.7740885416666667 |

### 7.2 Main6 干净 + 鲁棒矩阵（审计镜像）
数据来源：main6_matrix2_clean_robust.csv

| category | model | clean_R2 | clean_RMSE_kW | r2_sigma_030 | r2_sigma_060 | r2_spike_060 | rel_drop_060_pct | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| General DL | DLinear | 0.9379237294197084 | 6342.8623046875 | 0.2951500415802002 | -1.579047679901123 | 0.20584636926651 | 268.3556554481218 | 20260224T163948Z_main6_matrix_build_after_statsmodels_rerun |
| General DL | iTransformer | 0.9011505246162416 | 8004.04833984375 | 0.8937150239944458 | 0.8717493414878845 | 0.8609154224395752 | 3.2626273108862724 | 20260224T163948Z_main6_matrix_build_after_statsmodels_rerun |
| General DL | Mamba (auditable) | 0.7129279375076294 | 17640.982285575825 |  | 0.6941477060317993 | 0.7112981081008911 | 2.6342398000923843 | 20260224T163948Z_main6_matrix_build_after_statsmodels_rerun |
| Hybrid Opt. | VMD-PSO-LSTM | 0.9105949401855468 | 9844.831740563168 |  | 0.9057766199111938 | 0.9019150137901306 | 0.5291398031896845 | 20260224T163948Z_main6_matrix_build_after_statsmodels_rerun |
| Hybrid Opt. | CEEMDAN-WOA-GRU | 0.953388512134552 | 7108.4299250959775 |  | 0.9384608268737792 | 0.9278420805931092 | 1.565750485848728 | 20260224T163948Z_main6_matrix_build_after_statsmodels_rerun |
| Traditional | ARIMA (auditable) | 0.9301620961845448 | 10365.523839590212 | 0.921579638749407 | 0.8951699377795698 | 0.9191158877463884 | 3.761941982851433 | 20260224T163948Z_main6_matrix_build_after_statsmodels_rerun |
| Traditional | SVR (auditable) | 0.9101129642642014 | 11759.644320154111 | 0.8601274290576633 | 0.7268837572968938 | 0.8311609169097438 | 20.132578499773544 | 20260224T163948Z_main6_matrix_build_after_statsmodels_rerun |

## 八、公平性闭环与统计显著性

### 8.1 最小公平性对照（drift，2种子）
数据来源：fairness_minimal_aug_controls_drift_2seed.summary.csv

| model | training_mode | test_perturb | severity_eff | R2_mean | R2_std | RMSE_mean | RMSE_std | MAE_mean | MAE_std | n_seed | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear | clean_baseline | drift | 0.0 | 0.9379237294197083 | 0.0 | 6342.8623046875 | 0.0 | 4203.53466796875 | 0.0 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| DLinear | clean_baseline | drift | 0.6000000000000001 | 0.8729634881019592 | 0.0 | 9073.7509765625 | 0.0 | 7058.24755859375 | 0.0 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| DLinear | drift_augmented | drift | 0.0 | 0.9417210817337036 | 0.00019969176824461592 | 6145.793212890625 | 10.528226112656931 | 3981.9422607421875 | 4.459295718952566 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| DLinear | drift_augmented | drift | 0.6000000000000001 | 0.8998938798904419 | 0.0003722409660482161 | 8054.758056640625 | 14.975955387678955 | 6078.757080078125 | 12.664738203468628 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| TimeMixer | clean_baseline | drift | 0.0 | 0.8885634541511536 | 0.0 | 8498.384765625 | 0.0 | 6661.359375 | 0.0 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| TimeMixer | clean_baseline | drift | 0.6000000000000001 | 0.826326847076416 | 0.0 | 10609.3525390625 | 0.0 | 8724.7822265625 | 0.0 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| TimeMixer | drift_augmented | drift | 0.0 | 0.8797833025455475 | 0.00452096599921807 | 8826.05322265625 | 165.98986421371202 | 6852.950927734375 | 155.94363080932482 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| TimeMixer | drift_augmented | drift | 0.6000000000000001 | 0.8621975779533386 | 0.0027688793597716943 | 9450.1787109375 | 94.94703925741007 | 7449.236328125 | 17.54853967802122 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| iTransformer | clean_baseline | drift | 0.0 | 0.9011505246162415 | 0.0 | 8004.04833984375 | 0.0 | 6333.2939453125 | 0.0 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| iTransformer | clean_baseline | drift | 0.6000000000000001 | 0.8267310857772827 | 0.0 | 10596.998046875 | 0.0 | 8851.8974609375 | 0.0 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| iTransformer | drift_augmented | drift | 0.0 | 0.9029657542705536 | 0.006581356835521635 | 7927.93603515625 | 269.0113171355892 | 6235.275146484375 | 311.9290389235928 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |
| iTransformer | drift_augmented | drift | 0.6000000000000001 | 0.8772596120834351 | 0.008992620185678394 | 8916.0078125 | 326.8366314486375 | 7182.0791015625 | 368.5918393048759 | 2 | 20260301T164318Z_minimal_fairness_drift_2seed |

### 8.2 全故障公平性闭环：clean_trained vs fault_augmented
数据来源：fairness_closure_fullfault_hard_table.csv

| model | training_mode | n_seed | clean_R2_mean | clean_RMSE_mean | clean_AE_Q90_mean | gaussian_R2_mean | gaussian_RMSE_mean | gaussian_AE_Q90_mean | drift_R2_mean | drift_RMSE_mean | drift_AE_Q90_mean | quantization_R2_mean | quantization_RMSE_mean | quantization_AE_Q90_mean | missingness_R2_mean | missingness_RMSE_mean | missingness_AE_Q90_mean | avg_degradation_rmse_pct | avg_degradation_r2_drop_pct | worst_fault_AE_Q90_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear | clean_trained | 3 | 0.9432247690856457 | 6065.979654947917 | 9640.195638020834 | 0.24019533395767212 | 22190.193359375 | 35524.998046875 | 0.8000605503718058 | 11383.384440104166 | 18111.2908203125 | 0.936678779621919 | 6406.13623046875 | 10217.086979166668 | 0.9236601417263349 | 7033.82177734375 | 11328.494856770834 | 93.75894007723807 | 23.120324859438988 | 35524.998046875 |
| TimeMixer | clean_trained | 3 | 0.8727497185269991 | 9081.0361328125 | 13112.980794270836 | 0.7451838999986649 | 12848.90625 | 19952.967838541666 | 0.7239134709040324 | 13376.101888020834 | 18823.23932291667 | 0.8707762608925501 | 9151.1982421875 | 13177.424023437501 | 0.8680252283811569 | 9248.263997395834 | 13285.219140625 | 22.85323215148321 | 8.109532005239279 | 19952.967838541666 |
| Ultra-LSNT | clean_trained | 3 | 0.8618404964605967 | 9462.648111979166 | 14506.449934895834 | 0.8135411292314529 | 10992.4248046875 | 16695.233203125 | 0.7807923505703608 | 11918.438802083334 | 17702.446940104168 | 0.8610067715247472 | 9491.158528645834 | 14531.390429687503 | 0.8585051745176315 | 9576.189453125 | 14648.044921875 | 10.904885101829862 | 3.873029021131355 | 17702.446940104168 |
| iTransformer | clean_trained | 3 | 0.9074767058094343 | 7743.0634765625 | 11628.301953125001 | 0.8477625747521719 | 9933.029622395834 | 15358.29752604167 | 0.7828622857729594 | 11859.8603515625 | 18141.44720052084 | 0.9066339035828909 | 7778.2919921875 | 11652.290820312503 | 0.9028158336877823 | 7935.559733072917 | 11868.695833333333 | 21.09708327412461 | 5.230144261288067 | 18141.44720052084 |
| DLinear | fault_augmented | 3 | 0.9382064094146093 | 6328.260416666667 | 10093.326822916666 | 0.34711968898773193 | 20569.462239583332 | 32879.580078125 | 0.8330994794766108 | 10400.3837890625 | 16763.077083333334 | 0.932620736459891 | 6608.099934895833 | 10633.456966145835 | 0.9244839400053024 | 6995.836263020833 | 11226.206250000001 | 76.09539213711014 | 19.06575092762832 | 32879.580078125 |
| TimeMixer | fault_augmented | 3 | 0.875855008761088 | 8969.458658854166 | 13339.550716145832 | 0.8416698972384135 | 10129.427083333334 | 15023.653255208335 | 0.8342449814081192 | 10363.576822916666 | 15055.322526041668 | 0.8753350302577019 | 8988.232096354166 | 13356.7798828125 | 0.8731479669610659 | 9066.8701171875 | 13415.183203125001 | 7.448476989213357 | 2.255174076260185 | 15055.322526041668 |
| Ultra-LSNT | fault_augmented | 3 | 0.8479420244693756 | 9925.644856770834 | 15043.812174479166 | 0.817998523513476 | 10859.0263671875 | 16124.655859375003 | 0.811621348063151 | 11048.401692708334 | 16319.276171875003 | 0.8475470691919327 | 9938.528971354166 | 15049.520572916668 | 0.8445637226104736 | 10035.231119791666 | 15155.723111979167 | 5.489285362771643 | 2.0651885211944663 | 16319.276171875003 |
| iTransformer | fault_augmented | 3 | 0.8810667296250662 | 8779.478841145834 | 13219.93125 | 0.8579826106627783 | 9593.710286458334 | 14591.424283854169 | 0.8464618921279907 | 9975.312825520834 | 14897.016536458335 | 0.8806744267543157 | 8793.94140625 | 13238.253450520833 | 0.8776575153072675 | 8904.424479166666 | 13382.599544270835 | 6.121797910904639 | 1.744712438050018 | 14897.016536458335 |

### 8.3 统计显著性（Wilcoxon）
数据来源：fairness_closure_fullfault_significance.csv

| baseline | metric | n_pairs | mean_diff_ultra_minus_base | ci95_low | ci95_high | wilcoxon_stat | pvalue_one_sided_ultra_better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DLinear | R2 | 12 | 0.0711017046123743 | -0.056208616805573305 | 0.20999018612007297 | 33.0 | 0.68896484375 |
| DLinear | RMSE_gain(base-ultra) | 12 | 673.1485188802084 | -2066.5986531575522 | 3814.1542266845654 | 33.0 | 0.68896484375 |
| TimeMixer | R2 | 12 | -0.025666803121566772 | -0.02911959366562466 | -0.02243434712290764 | 0.0 | 1.0 |
| TimeMixer | RMSE_gain(base-ultra) | 12 | -833.2705078125 | -961.6912577311198 | -717.7756490071615 | 0.0 | 1.0 |
| iTransformer | R2 | 12 | -0.03526144536832968 | -0.03891013866911332 | -0.03178490869080027 | 0.0 | 1.0 |
| iTransformer | RMSE_gain(base-ultra) | 12 | -1153.4497884114583 | -1264.8710611979168 | -1046.519669596354 | 0.0 | 1.0 |

### 8.4 Diebold-Mariano检验（Wind CN）
数据来源：dm_complete_table.tex

~~~tex
\begin{table}[t]
\centering
\caption{Diebold--Mariano test on Wind (CN) (loss: squared error). Negative DM indicates Ultra-LSNT has smaller loss.}
\label{tab:dm_test}
\begin{tabular}{lcc}
\toprule
Comparison & DM statistic & $p$-value \\
\midrule
Ultra-LSNT vs PatchTST & -5.90 & $1.82 \times 10^{-9}$ \\
Ultra-LSNT vs TimeMixer & 3.45 & $2.83 \times 10^{-4}$ \\
Ultra-LSNT vs iTransformer & 5.12 & $3.07 \times 10^{-7}$ \\
\bottomrule
\end{tabular}
\end{table}
~~~

## 九、爬坡/尾部/平滑性与路由行为

### 9.1 爬坡-尾部-平滑性汇总（Ultra-LSNT vs DLinear）
数据来源：ramp_tail_smoothness_routing_ultra_dlinear.csv

| model | n_points | p95_abs_error_kw | max_abs_error_kw | ramp_score_f1 | ramp_tp | ramp_fp | ramp_fn | ramp_true_event_rate | ramp_pred_event_rate | pred_tv_kw | pred_tv_ratio_to_true | router_usage_var_mean | router_max_share_mean | router_entropy_norm_mean | ramp_window_hours | ramp_window_steps_15min | ramp_threshold_kw | ramp_threshold_pct_capacity | capacity_kw_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ultra-LSNT | 18111 | 11940.1748046875 | 34544.14453125 | 0.3543956043956044 | 129 | 253 | 217 | 0.01910863202076545 | 0.021096813387087868 | 8222569.28515625 | 0.2524735126420927 | 0.04413100229543193 | 0.5393562894625962 | 0.7510812491414316 | 1.0 | 4 | 6591.950000000001 | 0.05 | 131839.0 |
| DLinear | 18111 | 2484.4677734375 | 21479.015625 | 0.5939086294416244 | 234 | 208 | 112 | 0.01910863202076545 | 0.024410448997625227 | 29013519.33203125 | 0.8908584270722867 |  |  |  | 1.0 | 4 | 6591.950000000001 | 0.05 | 131839.0 |

路由原始轨迹文件：
- results/supplementary_evidence/routing_timestep_matrix_ultra_block1.csv
- results/supplementary_evidence/routing_timestep_topk_ultra_block1.csv

## 十、补充对比模型结果包

### 10.1 TSFM槽位结果
数据来源：tsfm_slots_windcn.csv

| category | model | clean_R2 | clean_RMSE_kW | clean_MAE_kW | R2_sigma_eff_060 | R2_spike_06 | drop_pct_sigma060 | R2_wind_us | R2_air_quality | R2_gefcom_load | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TSFM Slots | Chronos-T5-Mini | 0.9125753194688928 | 9740.122550270842 | 7312.858838212834 | 0.666382501811408 | 0.6554175256224304 | 26.977808012657636 |  |  |  | 20260227T161038Z_tsfm_slots_backfill |
| TSFM Slots | Moirai-Small | 0.910472013240847 | 9856.592534791416 | 7450.042844838971 | 0.6545707512530108 | 0.6401040578577212 | 28.106439106971397 |  |  |  | 20260227T161038Z_tsfm_slots_backfill |

### 10.2 扩展元启发式结果（干净场景）
数据来源：extended_metaheuristic_clean_windcn.csv

| dataset | model | R2 | RMSE | MAE | train_time_s |
| --- | --- | --- | --- | --- | --- |
| wind_cn | PSO-BPNN | 0.5857788920402527 | 21190.606220681842 | 16234.009765625 | 314.77666568756104 |
| wind_cn | GWO-SVR | -0.692350913159931 | 42832.40993882964 | 29985.787888788214 | 0.15999531745910645 |
| wind_cn | HPO-CNN-LSTM | 0.7573537826538086 | 16218.625342488185 | 12665.3486328125 | 238.75353527069092 |

### 10.3 扩展元启发式结果（鲁棒场景）
数据来源：extended_metaheuristic_robustness_windcn.csv

| dataset | model | fault_type | severity | sigma_eff | R2 | RMSE | MAE | R2_clean | relative_drop_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wind_cn | PSO-BPNN | gaussian | 0.6 | 0.6 | 0.5780383348464966 | 21387.683932581385 | 16303.5087890625 | 0.5857788920402527 | 1.3214127888396132 |
| wind_cn | GWO-SVR | gaussian | 0.6 | 0.6 | -0.6923237033235812 | 42832.06560493869 | 29985.16619578002 | -0.692350913159931 | -0.003930064340584766 |
| wind_cn | HPO-CNN-LSTM | gaussian | 0.6 | 0.6 | 0.7556825280189514 | 16274.383306288444 | 12676.5517578125 | 0.7573537826538086 | 0.22067026971183856 |
| wind_cn | PSO-BPNN | spike | 0.6 |  | -0.8353356122970581 | 44605.151989428305 | 34023.01171875 | 0.5857788920402527 | 242.60254571228742 |
| wind_cn | GWO-SVR | spike | 0.6 |  | -0.6923049766533058 | 42831.828621852306 | 29984.697205726778 | -0.692350913159931 | -0.006634858964146824 |
| wind_cn | HPO-CNN-LSTM | spike | 0.6 |  | 0.37524324655532837 | 26024.521359671537 | 18578.1640625 | 0.7573537826538086 | 50.45337395147612 |
| wind_cn | PSO-BPNN | drift | 0.6 |  | 0.09855407476425171 | 31260.585279229817 | 25144.017578125 | 0.5857788920402527 | 83.17555034770506 |
| wind_cn | GWO-SVR | drift | 0.6 |  | -0.6923077336915924 | 42831.86351182336 | 29984.747699270636 | -0.692350913159931 | -0.006236644961081064 |
| wind_cn | HPO-CNN-LSTM | drift | 0.6 |  | 0.6029006242752075 | 20748.030460744943 | 16875.703125 | 0.7573537826538086 | 20.393792427785193 |

## 十一、证据可追溯与审计

### 11.1 证据清单（manifest）
数据来源：evidence_manifest.csv

| claim_id | file | rows | generated_run_id |
| --- | --- | --- | --- |
| core1_protocol | split_manifest_80_20_unified.json;deep_sota_gaussian_windcn.csv;structured_scada_fault_robustness_windcn.csv | 37 | 20260224T091423Z_full_revision_run |
| core2_clean | core2_clean.csv | 9 | 20260224T091423Z_full_revision_run |
| core3_robustness | core3_robustness.csv | 21 | 20260224T091423Z_full_revision_run |
| core4_hardware | core4_hardware.csv | 8 | 20260224T091423Z_full_revision_run |
| core5_ablation_physics | pi_moe_full_vs_ultra_clean_sigma06_windcn.csv;lambda_robustness_sigma_eff_compare.csv;tau_sweep_ultra_pareto.csv;joint_sensitivity_soa_windcn.csv;joint_sensitivity_pareto_frontier.csv | 73 | 20260224T091423Z_full_revision_run |
| core6_dispatch | dispatch_rolling_with_ultra_scaled_summary.csv;dispatch_network_ieee24_dcopf_wmax260_aggregate.csv;dispatch_network_ieee24_dcopf_wmax280_aggregate.csv | 12 | 20260224T091423Z_full_revision_run |
| core7_transfer | table_multi_domain.csv | 4 | 20260224T091423Z_full_revision_run |

### 11.2 验证与冲突消解日志
- verification_report_main_tables_20260303.md
- conflict_resolution_log_20260303.md
- conflict_resolution_round2_20260303.csv
- dedupe_applied_identical_20260303.csv

## 十二、基于当前规范证据的最终结论
1. 论文正文主表数值已与 `results/supplementary_evidence` 规范证据文件对齐。
2. 在高斯强扰动下，Ultra-LSNT 仍保持正向技能（R2=0.7263）；FG-MoE 将 over-rated 违例率从 0.0656% 降到 0.0023%。
3. 在 mapped 调度模式下，Ultra-LSNT 在 L1 成本与弃风上相对 DLinear 有明显改进；iTransformer 在 mapped 汇总指标上领先。
4. 报告已包含公平性闭环与显著性表，可直接用于答辩审计，并支撑后续 baseline 对称扩展。

## 十三、规范证据文件完整索引
当前 `results/supplementary_evidence` 目录下全部文件如下：

- arima_clean_windcn.csv
- arima_latency_windcn.csv
- arima_meta.json
- arima_robustness_windcn.csv
- arm_revision_eval.meta.json
- arm_revision_latency_energy.csv
- arm_revision_robustness_smoke.csv
- core2_clean.csv
- core3_robustness.csv
- core4_hardware.csv
- cpu_batch1_latency_table.csv
- cpu_batch1_ultra_family_latency.csv
- deep_sota_gaussian_windcn.csv
- deep_sota_gaussian_windcn.meta.json
- dispatch_closure_mapping_decision.meta.json
- dispatch_closure_mapping_decision_aggregate.csv
- dispatch_closure_mapping_decision_daily.csv
- dispatch_closure_mapping_decision_hourly.csv
- dispatch_closure_mapping_decision_ranking_stability.csv
- dispatch_mapping_daylevel_consistency_wmax260_280_300.csv
- dispatch_mapping_daylevel_consistency_wmax260_280_300.meta.json
- dispatch_mapping_sensitivity_wmax260_280.csv
- dispatch_mapping_sensitivity_wmax260_280.meta.json
- dispatch_mapping_sensitivity_wmax260_280_300.csv
- dispatch_mapping_sensitivity_wmax260_280_300.meta.json
- dispatch_network_ieee24_dcopf_aggregate.csv
- dispatch_network_ieee24_dcopf_daily.csv
- dispatch_network_ieee24_dcopf_main_aggregate.csv
- dispatch_network_ieee24_dcopf_main_daily.csv
- dispatch_network_ieee24_dcopf_main_meta.json
- dispatch_network_ieee24_dcopf_meta.json
- dispatch_network_ieee24_dcopf_wmax260_aggregate.csv
- dispatch_network_ieee24_dcopf_wmax260_daily.csv
- dispatch_network_ieee24_dcopf_wmax260_meta.json
- dispatch_network_ieee24_dcopf_wmax280_aggregate.csv
- dispatch_network_ieee24_dcopf_wmax280_daily.csv
- dispatch_network_ieee24_dcopf_wmax280_meta.json
- dispatch_network_ieee24_dcopf_wmax300_aggregate.csv
- dispatch_network_ieee24_dcopf_wmax300_daily.csv
- dispatch_network_ieee24_dcopf_wmax300_meta.json
- dispatch_raw_vs_mapped_summary.csv
- dispatch_rolling_with_ultra_daily.csv
- dispatch_rolling_with_ultra_scaled_aggregate.csv
- dispatch_rolling_with_ultra_scaled_daily.csv
- dispatch_rolling_with_ultra_scaled_meta.json
- dispatch_rolling_with_ultra_scaled_summary.csv
- dispatch_rolling_with_ultra_summary.csv
- dlinear_noise_sanity_multiseed.csv
- dlinear_predictions_full.npz
- evidence_manifest.csv
- extended_metaheuristic_best_configs.csv
- extended_metaheuristic_clean_windcn.csv
- extended_metaheuristic_latency_windcn.csv
- extended_metaheuristic_meta.json
- extended_metaheuristic_robustness_windcn.csv
- extended_metaheuristic_search_trace.csv
- fairness_closure_fullfault_hard_table.csv
- fairness_closure_fullfault_meta.json
- fairness_closure_fullfault_raw.csv
- fairness_closure_fullfault_significance.csv
- fairness_closure_fullfault_summary.csv
- fairness_minimal_aug_controls.csv
- fairness_minimal_aug_controls.meta.json
- fairness_minimal_aug_controls_drift_2seed.csv
- fairness_minimal_aug_controls_drift_2seed.meta.json
- fairness_minimal_aug_controls_drift_2seed.summary.csv
- fgmoe_clip_only_control_windcn.csv
- fgmoe_clip_only_control_windcn.meta.json
- fig_joint_nkt_pareto_3d.pdf
- fig_lambda2_tradeoff_anchor.pdf
- fig_soa_3d_pareto.pdf
- joint_lambda2_tradeoff_anchor.csv
- joint_parameter_full_windcn.csv
- joint_parameter_meta.json
- joint_sensitivity_meta.json
- joint_sensitivity_pareto_frontier.csv
- joint_sensitivity_soa_windcn.csv
- joint_structural_nkt_windcn.csv
- joint_structural_pareto_windcn.csv
- lambda_robustness_sigma_eff_compare.csv
- lambda_robustness_sigma_eff_compare.meta.json
- latency_arm_full_backfill.csv
- lite_accuracy_for_pareto.csv
- lite_robustness_windcn.csv
- main4_matrix1_protocol.csv
- main4_matrix2_perf_robust_transfer.csv
- main4_matrix2_perf_robust_transfer_mainclaim.csv
- main4_matrix3_hardware_safety_joint.csv
- main4_matrix4_dispatch_value.csv
- main4_matrix_manifest.csv
- main6_matrix2_clean_robust.csv
- main6_matrix3_hardware_ablation.csv
- main6_matrix4_joint_parameter.csv
- main6_matrix_manifest.csv
- mamba_auditable_checkpoint.pth
- mamba_clean_windcn.csv
- mamba_latency_windcn.csv
- mamba_meta.json
- mamba_robustness_windcn.csv
- minimal_vmd_ceemdan_clean_windcn.csv
- minimal_vmd_ceemdan_latency_windcn.csv
- minimal_vmd_ceemdan_meta.json
- minimal_vmd_ceemdan_robustness_windcn.csv
- minimal_vmd_ceemdan_search_trace.csv
- missingness_imputation_ultra_dlinear.csv
- missingness_imputation_ultra_dlinear_summary.csv
- multi_domain_transfer_matrix.csv
- multi_domain_transfer_matrix_all_models.csv
- noise_calibration_diff_proxy_summary.json
- noise_calibration_diff_proxy_windcn.csv
- persistence_gaussian_windcn.csv
- persistence_gaussian_windcn.meta.json
- persistence_two_point_windcn.csv
- physics_closure.csv
- physics_closure.meta.json
- pi_moe_curve_params_windcn.csv
- pi_moe_evidence_meta.json
- pi_moe_full_checkpoint.pth
- pi_moe_full_structured_scada_fault_robustness_windcn.csv
- pi_moe_full_training_history.csv
- pi_moe_full_training_meta.json
- pi_moe_full_vs_ultra_clean_sigma06_windcn.csv
- pi_moe_physics_variable_availability_windcn.csv
- pi_moe_vs_ultra_clean_sigma06_windcn.csv
- probabilistic_gaussian_minimal_ultra_dlinear.csv
- ramp_tail_smoothness_routing_ultra_dlinear.csv
- robust_two_point_windcn.csv
- rolling_dispatch_multiday_daily.csv
- rolling_dispatch_multiday_summary.csv
- rolling_dispatch_multiday_summary.json
- routing_timestep_matrix_ultra_block1.csv
- routing_timestep_topk_ultra_block1.csv
- ssa_elm_clean_windcn.csv
- ssa_elm_latency_windcn.csv
- ssa_elm_meta.json
- ssa_elm_robustness_windcn.csv
- ssa_elm_search_trace.csv
- structured_scada_fault_robustness_windcn.csv
- svr_clean_windcn.csv
- svr_latency_windcn.csv
- svr_meta.json
- svr_robustness_windcn.csv
- tau_sweep_ultra_pareto.csv
- tau_sweep_ultra_pareto.pdf
- tau_sweep_ultra_pareto.png
- tau_sweep_ultra_pareto.svg
- tsfm_slots_windcn.csv
- ultra_lambda005_finetuned_from_main.pth
- ultra_metrics_main_fulltest.json
- ultra_predictions_main_fulltest.npz
- validate_core_matrix_report.txt
- y_true_y_pred_ultra_dlinear_first_month_15min.csv
- y_true_y_pred_ultra_dlinear_full_15min.csv
