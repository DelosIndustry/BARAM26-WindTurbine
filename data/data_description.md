# 데이터 명세서

## 1. 제공 파일

참가자에게 제공되는 파일은 아래와 같습니다.

| 구분 | 파일 | 행 수 | 내용 |
|---|---|---:|---|
| 학습 데이터 | `train/ldaps_train.csv` | 420,864 | 학습 기간의 LDAPS 기상 예보 데이터 |
| 학습 데이터 | `train/gfs_train.csv` | 236,736 | 학습 기간의 GFS 기상 예보 데이터 |
| 학습 데이터 | `train/train_labels.csv` | 26,304 | 학습 기간의 KPX 그룹별 실제 발전량 |
| 학습 데이터 | `train/scada_vestas_train.csv` | 157,819 | VESTAS 터빈 SCADA 실측 데이터 |
| 학습 데이터 | `train/scada_unison_train.csv` | 105,264 | UNISON 터빈 SCADA 실측 데이터 |
| 평가 데이터 | `test/ldaps_test.csv` | 140,160 | 평가 기간의 LDAPS 기상 예보 데이터 |
| 평가 데이터 | `test/gfs_test.csv` | 78,840 | 평가 기간의 GFS 기상 예보 데이터 |
| 제출 양식 | `sample_submission.csv` | 8,760 | 참가자가 예측값을 채워 제출할 CSV 양식 |
| 메타 정보 | `info.xlsx` | - | KPX 그룹, 터빈, 설비용량, 위치 정보 |
| 데이터 설명 | `data_description.md` | - | 제공 데이터 파일과 컬럼에 대한 설명 문서 |

모든 CSV 파일은 `UTF-8 with BOM`(`utf-8-sig`)으로 저장되어 있습니다.

시간은 모두 KST 기준이며, 형식은 `YYYY-MM-DD HH:MM:SS`입니다.

## 2. 예측 목표

참가자는 2025년 전체 기간에 대해 3개 KPX 그룹의 시간별 발전량을 예측합니다.

| 컬럼 | 의미 | 설비용량 |
|---|---|---:|
| `kpx_group_1` | KPX 그룹 1 발전량 | 21.6 MW |
| `kpx_group_2` | KPX 그룹 2 발전량 | 21.6 MW |
| `kpx_group_3` | KPX 그룹 3 발전량 | 21.0 MW |

예측해야 하는 발전량 단위는 `kWh`입니다.

설비용량 단위는 `MW`입니다. 평가에서 발전량과 설비용량을 함께 사용할 때는 시간 단위 발전량 기준에 맞춰 설비용량을 `kWh`로 환산해야 합니다.

1시간 기준 환산값은 아래와 같습니다.

| 컬럼 | 설비용량 | 1시간 기준 설비용량 |
|---|---:|---:|
| `kpx_group_1` | 21.6 MW | 21,600 kWh |
| `kpx_group_2` | 21.6 MW | 21,600 kWh |
| `kpx_group_3` | 21.0 MW | 21,000 kWh |

## 3. 기간 요약

| 파일 | 기간 |
|---|---|
| `ldaps_train.csv`, `gfs_train.csv` | 2022-01-01 01:00:00 ~ 2025-01-01 00:00:00 |
| `train_labels.csv` | 2022-01-01 01:00:00 ~ 2025-01-01 00:00:00 |
| `ldaps_test.csv`, `gfs_test.csv` | 2025-01-01 01:00:00 ~ 2026-01-01 00:00:00 |
| `sample_submission.csv` | 2025-01-01 01:00:00 ~ 2026-01-01 00:00:00 |

## 4. `info.xlsx`

KPX 그룹과 풍력발전기 메타 정보를 담은 파일입니다.

이 파일은 KPX 원천데이터 파일의 `info` 시트 내용을 제공합니다.

| 컬럼 | 의미 |
|---|---|
| `단계` | 풍력발전단지 단계 구분 |
| `명칭` | 발전단지 또는 발전기 위치 명칭 |
| `제작사` | 풍력터빈 제작사 |
| `모델명` | 풍력터빈 모델명 |
| `호기` | 터빈 번호 |
| `좌표(Google)` | Google 기준 터빈 위치 좌표 |
| `KPX그룹` | 해당 터빈이 속한 KPX 평가 그룹 |
| `Hub Height(m)` | 허브 높이 |
| `Rotor Diameter(m)` | 로터 직경 |
| `설비용량(MW)` | 개별 터빈 설비용량 |
| `그룹설비용량(MW)` | KPX 그룹별 총 설비용량 |

## 5. `train_labels.csv`

학습용 실제 발전량 데이터입니다.

| 컬럼 | 의미 |
|---|---|
| `kst_dtm` | 실제 발전량 집계 구간의 종료 시각입니다. |
| `kpx_group_1` | KPX 그룹 1 실제 발전량입니다. 단위는 `kWh`입니다. |
| `kpx_group_2` | KPX 그룹 2 실제 발전량입니다. 단위는 `kWh`입니다. |
| `kpx_group_3` | KPX 그룹 3 실제 발전량입니다. 단위는 `kWh`입니다. |

그룹별 Label 제공 기간은 아래와 같습니다.

| 컬럼 | 제공 기간 | 비고 |
|---|---|---|
| `kpx_group_1` | 2022년-2024년 | 2022년부터 값이 제공됩니다. |
| `kpx_group_2` | 2022년-2024년 | 2022년부터 값이 제공됩니다. |
| `kpx_group_3` | 2023년-2024년 | 2022년 구간은 빈칸입니다. |

`kst_dtm`은 1시간 단위입니다.

## 6. 기상 예보 데이터 공통 설명

대상 파일:

- `ldaps_train.csv`
- `ldaps_test.csv`
- `gfs_train.csv`
- `gfs_test.csv`

기상 예보 데이터는 매일 `09:00 KST`에 초기화된 예보자료를 사용합니다.

이 예보자료 중 다음날 `01:00:00`부터 그 다음날 `00:00:00`까지의 24시간 예보값만 추출해 날짜별로 이어 붙였습니다.

각 예보자료는 해당일 `13:00 KST`부터 사용 가능한 것으로 간주합니다.

| 컬럼 | 의미 |
|---|---|
| `forecast_kst_dtm` | 예보 대상 시각입니다. 발전량을 예측해야 하는 시각과 대응됩니다. |
| `data_available_kst_dtm` | 해당 예보 데이터가 사용 가능해진 시각입니다. |
| `grid_id` | 기상 격자 ID입니다. |
| `latitude` | 격자 위도입니다. |
| `longitude` | 격자 경도입니다. |

예:

| `forecast_kst_dtm` | `data_available_kst_dtm` |
|---|---|
| `2025-01-01 01:00:00` | `2024-12-31 13:00:00` |
| `2025-01-01 02:00:00` | `2024-12-31 13:00:00` |
| `2025-01-02 00:00:00` | `2024-12-31 13:00:00` |
| `2025-01-02 01:00:00` | `2025-01-01 13:00:00` |

즉 `01:00:00`부터 다음날 `00:00:00`까지의 24개 예보 대상 시각은 같은 `data_available_kst_dtm`을 갖습니다.

## 7. `ldaps_train.csv`, `ldaps_test.csv`

LDAPS 기상 예보 데이터입니다. 약 1.5 km 공간해상도의 16개 격자에 대해 제공됩니다.

`forecast_kst_dtm` 하나당 `grid_id` 16개 행이 존재합니다.

| 컬럼 | 의미 |
|---|---|
| `forecast_kst_dtm` | 예보 대상 시각 |
| `data_available_kst_dtm` | 예보 데이터 사용 가능 시각 |
| `grid_id` | LDAPS 격자 ID |
| `latitude` | 격자 위도 |
| `longitude` | 격자 경도 |
| `heightAboveGround_10_10u` | 지상 10 m U 성분 바람 |
| `heightAboveGround_10_10v` | 지상 10 m V 성분 바람 |
| `heightAboveGround_50_50MUmax` | 지상 50 m U 성분 바람 최댓값 |
| `heightAboveGround_50_50MUmin` | 지상 50 m U 성분 바람 최솟값 |
| `heightAboveGround_50_50MVmax` | 지상 50 m V 성분 바람 최댓값 |
| `heightAboveGround_50_50MVmin` | 지상 50 m V 성분 바람 최솟값 |
| `heightAboveGround_5_XBLWS` | 지상 5 m X 방향 경계층 바람 |
| `heightAboveGround_5_YBLWS` | 지상 5 m Y 방향 경계층 바람 |
| `heightAboveGround_2_t` | 지상 2 m 기온 |
| `heightAboveGround_2_dpt` | 지상 2 m 이슬점온도 |
| `heightAboveGround_2_r` | 지상 2 m 상대습도 |
| `heightAboveGround_2_q` | 지상 2 m 비습 |
| `surface_0_sp` | 지표면 기압 |
| `meanSea_0_prmsl` | 해면기압 |
| `etc_0_blh` | 경계층 높이 |
| `surface_0_NDNSW` | 지표면 순 하향 단파복사 |
| `surface_0_NDNLW` | 지표면 순 하향 장파복사 |
| `heightAboveGround_2_SWDIR` | 지상 2 m 직접 단파복사 |
| `heightAboveGround_2_SWDIF` | 지상 2 m 산란 단파복사 |
| `etc_0_hcc` | 상층운량 |
| `etc_0_mcc` | 중층운량 |
| `etc_0_lcc` | 하층운량 |
| `etc_0_VLCDC` | 매우 낮은 층 운량 |
| `surface_0_avg_lsprate` | 지표면 평균 대규모 강수율 |
| `surface_0_lssrate` | 지표면 대규모 강설률 |
| `surface_0_ncpcp` | 지표면 비대류성 강수량 |
| `surface_0_snol` | 지표면 적설 관련 변수 |
| `surface_0_SNOM` | 지표면 융설량 |
| `surface_0_lsm` | 육지/해양 마스크 |
| `surface_0_h` | 지표면 고도 |

## 8. `gfs_train.csv`, `gfs_test.csv`

GFS 기상 예보 데이터입니다. 약 0.25도 공간해상도의 9개 격자에 대해 제공됩니다.

`forecast_kst_dtm` 하나당 `grid_id` 9개 행이 존재합니다.

| 컬럼 | 의미 |
|---|---|
| `forecast_kst_dtm` | 예보 대상 시각 |
| `data_available_kst_dtm` | 예보 데이터 사용 가능 시각 |
| `grid_id` | GFS 격자 ID |
| `latitude` | 격자 위도 |
| `longitude` | 격자 경도 |
| `heightAboveGround_10_10u` | 지상 10 m U 성분 바람 |
| `heightAboveGround_10_10v` | 지상 10 m V 성분 바람 |
| `heightAboveGround_80_u` | 지상 80 m U 성분 바람 |
| `heightAboveGround_80_v` | 지상 80 m V 성분 바람 |
| `heightAboveGround_100_100u` | 지상 100 m U 성분 바람 |
| `heightAboveGround_100_100v` | 지상 100 m V 성분 바람 |
| `heightAboveGround_2_2t` | 지상 2 m 기온 |
| `heightAboveGround_2_2d` | 지상 2 m 이슬점온도 |
| `heightAboveGround_2_2r` | 지상 2 m 상대습도 |
| `heightAboveGround_2_2sh` | 지상 2 m 비습 |
| `planetaryBoundaryLayer_0_u` | 행성경계층 U 성분 바람 |
| `planetaryBoundaryLayer_0_v` | 행성경계층 V 성분 바람 |
| `planetaryBoundaryLayer_0_VRATE` | 행성경계층 수직 속도 |
| `surface_0_dswrf` | 지표면 하향 단파복사 플럭스 |
| `surface_0_dlwrf` | 지표면 하향 장파복사 플럭스 |
| `surface_0_prate` | 지표면 강수율 |
| `surface_0_tp` | 지표면 총강수량 |
| `surface_0_sp` | 지표면 기압 |
| `meanSea_0_prmsl` | 해면기압 |
| `surface_0_gust` | 지표면 돌풍 |
| `lowCloudLayer_0_lcc` | 하층운량 |
| `middleCloudLayer_0_mcc` | 중층운량 |
| `highCloudLayer_0_hcc` | 상층운량 |
| `atmosphere_0_tcc` | 전운량 |
| `isobaricInhPa_850_t` | 850 hPa 기온 |
| `isobaricInhPa_850_u` | 850 hPa U 성분 바람 |
| `isobaricInhPa_850_v` | 850 hPa V 성분 바람 |
| `isobaricInhPa_850_r` | 850 hPa 상대습도 |
| `isobaricInhPa_700_t` | 700 hPa 기온 |
| `isobaricInhPa_700_u` | 700 hPa U 성분 바람 |
| `isobaricInhPa_700_v` | 700 hPa V 성분 바람 |
| `isobaricInhPa_500_gh` | 500 hPa 지위고도 |
| `isobaricInhPa_500_t` | 500 hPa 기온 |
| `isobaricInhPa_500_u` | 500 hPa U 성분 바람 |
| `isobaricInhPa_500_v` | 500 hPa V 성분 바람 |

## 9. `scada_vestas_train.csv`

VESTAS 터빈 SCADA 학습 데이터입니다.

| 컬럼 | 의미 |
|---|---|
| `kst_dtm` | SCADA 계측 시각 |
| `vestas_wtg01_power_kw10m` ~ `vestas_wtg12_power_kw10m` | VESTAS 1-12호기 10분 단위 power 값입니다. 원천 컬럼명 기준 단위는 `kW10m`입니다. |
| `vestas_wtg01_ws` ~ `vestas_wtg12_ws` | VESTAS 1-12호기 풍속입니다. |
| `vestas_wtg01_wd` ~ `vestas_wtg12_wd` | VESTAS 1-12호기 풍향입니다. |

## 10. `scada_unison_train.csv`

UNISON 터빈 SCADA 학습 데이터입니다.

| 컬럼 | 의미 |
|---|---|
| `kst_dtm` | SCADA 계측 시각 |
| `unison_wtg01_power_kw10m` ~ `unison_wtg05_power_kw10m` | UNISON 1-5호기 10분 단위 power 값입니다. 원천 컬럼명 기준 단위는 `kW10m`입니다. |
| `unison_wtg01_ws` ~ `unison_wtg05_ws` | UNISON 1-5호기 풍속입니다. |
| `unison_wtg01_wd` ~ `unison_wtg05_wd` | UNISON 1-5호기 풍향입니다. |

## 11. `sample_submission.csv`

제출 양식 파일입니다.

| 컬럼 | 의미 |
|---|---|
| `forecast_id` | 제출 행을 구분하는 고유 ID입니다. 평가 시 매칭 키로 사용됩니다. |
| `forecast_kst_dtm` | 예측 대상 시각입니다. |
| `kpx_group_1` | KPX 그룹 1 발전량 예측값을 입력합니다. 단위는 `kWh`입니다. |
| `kpx_group_2` | KPX 그룹 2 발전량 예측값을 입력합니다. 단위는 `kWh`입니다. |
| `kpx_group_3` | KPX 그룹 3 발전량 예측값을 입력합니다. 단위는 `kWh`입니다. |

참가자는 `forecast_id`와 `forecast_kst_dtm`을 변경하지 말고, `kpx_group_1`, `kpx_group_2`, `kpx_group_3` 값만 채워 제출해야 합니다.

## 12. 주의사항

평가 기간의 실제 발전량, SCADA, 비공개 운영 데이터, 사후 보정자료, 재분석자료 등은 예측에 사용할 수 없습니다.

Excel에서 CSV 파일을 열고 다시 저장하면 시간 문자열 형식이 바뀔 수 있습니다. 제출 파일은 코드로 생성하는 것을 권장합니다.
