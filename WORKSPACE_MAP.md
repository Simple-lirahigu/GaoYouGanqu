# Workspace Map

## scripts

- `scripts/gaoyou_irrigation_2020.js`：高邮灌区 2020 年灌溉稻田初步识别 GEE JavaScript。
- `scripts/gaoyou_landcover_rf_2020.js`：高邮地区 2020 年 S1/S2 多时相随机森林土地利用分类、空间验证和耕地面积统计。

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
