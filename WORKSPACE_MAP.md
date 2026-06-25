# Workspace Map

## scripts

- `scripts/gaoyou_irrigation_2020.js`：高邮灌区 2020 年灌溉稻田初步识别 GEE JavaScript。
- `scripts/gaoyou_landcover_rf_2020.js`：高邮地区 2020 年 S1/S2 多时相随机森林土地利用分类、空间验证和耕地面积统计。
- `scripts/gaoyou_landcover_hierarchical_rf_2020.js`：增强版分层随机森林；先识别永久水体，再分类林地、耕地、建筑和其他，重点降低耕地误判水体。

## 项目管理

- `VERSIONING.md`：Git提交、实验分支、版本标签和GitHub Release规则。
- `HISTORY_LOG.md`：按日期记录项目有效修改。

## 数据资产

- 研究区：`projects/ee-yangsimple237/assets/GYBJ`
- 2020 年土地利用：`projects/ee-yangsimple237/assets/2020tudi`

## 默认输出

- Google Drive 文件夹：`GEE_Gaoyou_Irrigation`
- 灌溉稻田分类影像：`gaoyou_irrigated_rice_2020_10m.tif`
- 诊断特征影像：`gaoyou_irrigation_diagnostics_2020.tif`
- 面积统计：`gaoyou_irrigated_rice_area_2020.csv`

## 土地利用随机森林默认输出

- Google Drive 文件夹：`GEE_Gaoyou_Landcover_RF`
- 五类分类影像：`gaoyou_landcover_rf_2020_10m.tif`
- 耕地影像：`gaoyou_cropland_rf_2020_10m.tif`
- 分类置信度：`gaoyou_landcover_rf_confidence_2020_10m.tif`
- 面积、精度、变量重要性 CSV 和训练/调参/测试样本 SHP

## 增强版分层随机森林默认输出

- Google Drive 文件夹：`GEE_Gaoyou_Hierarchical_RF`
- 五类分层分类、耕地、永久水体概率和置信度 GeoTIFF
- 独立测试精度、面积和两阶段变量重要性 CSV
- 独立测试样本 GeoJSON
- 水体训练、调参、独立测试人工质检点 SHP，以及合并质检表 CSV
- 合并水体样本点 SHP：`samp_type=1/2/3` 分别表示训练、调参和独立测试点
- SHP内含人工核查字段：`man_label`、`qa_stat`、`qa_note`、`reviewer`、`rev_date`
- 水体样本类型 GeoTIFF：`0=无样本，1=训练点，2=调参点，3=独立测试点`
- 二值水体预测 GeoTIFF：`1=水体，0=非水体`；同时保留水体概率 GeoTIFF
