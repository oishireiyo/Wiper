# Standard python modules
import os
import sys
import math
from typing import Union, Any

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Handmade modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FaceExpressionWithPyFeat.src.facial_expression import FacialExpressionAnalysis

class Wiper(object):
  def __init__(
      self,
      video_paths: list[str], # mp4などの動画ファイルのパス
      weights: list[float], # どれだけ対応する動画ファイルを重要視するか
      interval_in_frame: int=30, # 表情検出、角度検出などを何フレームおきに実行するか
      csv_paths: Union[list[str], None]=None, # 前回結果を保存したCSVファイルがある場合は指定して時短
      criteria: Union[list[tuple[str, str]], None]=None, # どのような条件でワイプ動画を選定するか
      output_video_name: str='output.mp4',
      output_video_size: tuple[int, int]=(400, 400),
  ) -> None:
    self.video_paths = video_paths
    self.weights = weights
    self.interval_in_frame = interval_in_frame
    self.csv_paths = [] if csv_paths is None else csv_paths
    self.dataframes = []
    self.criteria = criteria
    self.output_video_name = output_video_name
    self.output_video_size = output_video_size

  def convert_videos_to_dataframe(self) -> None:
    for video_path in self.video_paths:
      parser = FacialExpressionAnalysis()
      parser.set_detector()
      results = parser.detect_video_with_images(
        video_path=video_path,
        interval_in_frame=self.interval_in_frame
      )
      csv_path = '../deliverables/' + os.path.splitext(os.path.basename(video_path))[0] + '.csv'
      parser.save_as_csv(
        results=results,
        csvfile=csv_path
      )
      self.csv_paths.append(csv_path)

  def get_dataframe(self, csv_path: str) -> pd.DataFrame:
    parser = FacialExpressionAnalysis()
    return parser.read_results(csvfile=csv_path)

  def set_dataframes(self) -> None:
    for csv_path in self.csv_paths:
      dataframe = self.get_dataframe(csv_path=csv_path)
      self.dataframes.append(dataframe)

  def set_criteria(self, criteria: list[tuple[str, str]]) -> None:
    self.criteria = criteria

  def get_criteria_str(self) -> str:
    # 入力のcriteriaは以下の形式を想定
    #   criteria = [
    #     ('happiness', '> 0.9'),
    #     ('Pitch', '< 15.0'),
    #   ]
    criteria_str =' and '.join([f'{criterion[0]} {criterion[1]}' for criterion in self.criteria])
    return criteria_str

  def get_fulfil_records(self, df: pd.DataFrame) -> pd.DataFrame:
    criteria = self.get_criteria_str()
    return df.query(criteria)

  def add_columns_with_criteria(self, df: pd.DataFrame, column_name: str, true_val: Any, false_val: Any):
    df[column_name] = True
    for criterion in self.criteria:
      df[column_name] = [true_val if eval(f'{x} {criterion[1]} and {y}') else false_val for x, y in zip(df[criterion[0]], df[column_name])]

  def add_boolean_columns_with_criteria(self, df: pd.DataFrame, column_name: str):
    self.add_columns_with_criteria(df=df, column_name=column_name, true_val=True, false_val=False)

  def add_bit_columns_with_criteria(self, df: pd.DataFrame, column_name: str):
    self.add_columns_with_criteria(df=df, column_name=column_name, true_val=1, false_val=0)

  def add_float_columns_with_criteria(self, df: pd.DataFrame, column_name: str, true_val: float, false_val: float):
    self.add_columns_with_criteria(df=df, column_name=column_name, true_val=true_val, false_val=false_val)

  def add_columns_with_moving_sum(self, df: pd.DataFrame, window_size: int, column: str):
    df[f'{column}_move_sum_{window_size}'] = df[column].rolling(window=window_size, min_periods=1).sum().shift(-window_size+1)
    df[f'{column}_move_sum_{window_size}'] = df[f'{column}_move_sum_{window_size}'].fillna(0)

  def add_columns_with_moving_mean(self, df: pd.DataFrame, window_size: int, column: str):
    df[f'{column}_move_mean_{window_size}'] = df[column].rolling(window=window_size, min_periods=1).mean().shift(-window_size+1)
    df[f'{column}_move_mean_{window_size}'] = df[f'{column}_move_mean_{window_size}'].fillna(0)

  def make_wipe(self, window_size: int=10):
    column = 'fulfil'
    for df, video_path, weight in zip(self.dataframes, self.video_paths, self.weights):
      # はじめにどのレコードが条件を満たすかをbit値で表現する
      self.add_float_columns_with_criteria(df=df, column_name=column, true_val=weight, false_val=0.0)

      # 各レコードに対し、window_sizeの移動和をとる
      self.add_columns_with_moving_sum(df=df, window_size=window_size, column=column)

      # わかりやすいように列名を変えておく
      df[video_path] = df[f'{column}_move_sum_{window_size}']
      for facecolumn in ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']:
        df[f'{video_path}_{facecolumn}'] = df[facecolumn]

    # データフレームの結合
    for i_df in range(len(self.dataframes)):
      if i_df == 0:
        merged_df = pd.merge(
          self.dataframes[i_df][[
            'frame',
            self.video_paths[i_df],
            f'{self.video_paths[i_df]}_FaceRectX',
            f'{self.video_paths[i_df]}_FaceRectY',
            f'{self.video_paths[i_df]}_FaceRectWidth',
            f'{self.video_paths[i_df]}_FaceRectHeight',
          ]],
          self.dataframes[i_df+1][[
            'frame',
            self.video_paths[i_df+1],
            f'{self.video_paths[i_df+1]}_FaceRectX',
            f'{self.video_paths[i_df+1]}_FaceRectY',
            f'{self.video_paths[i_df+1]}_FaceRectWidth',
            f'{self.video_paths[i_df+1]}_FaceRectHeight',
          ]],
          on='frame',
        )
      else:
        if i_df < len(self.dataframes)-1:
          merged_df = pd.merge(
            merged_df,
            self.dataframes[i_df+1][[
              'frame',
              self.video_paths[i_df+1],
              f'{self.video_paths[i_df+1]}_FaceRectX',
              f'{self.video_paths[i_df+1]}_FaceRectY',
              f'{self.video_paths[i_df+1]}_FaceRectWidth',
              f'{self.video_paths[i_df+1]}_FaceRectHeight',
            ]],
            on='frame',
          )

    # wipe作成に必要なデータフレーム
    def condition_based_value(row, facecolumn):
      video_name_from = row['video_name_from']
      return row[f'{video_name_from}_{facecolumn}']

    merged_df['video_name_from'] = merged_df[self.video_paths].idxmax(axis=1)
    wipe_df = pd.DataFrame()
    wipe_df['frame'] = merged_df['frame']
    wipe_df['video_name_from'] = merged_df['video_name_from']
    for facecolumn in ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']:
      wipe_df[facecolumn] = merged_df.apply(condition_based_value, args=(facecolumn, ), axis=1)

    # ビデオを出力
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = self.interval_in_frame
    wipe_video = cv2.VideoWriter(self.output_video_name, fourcc, fps, self.output_video_size)
    video_w, video_h = self.output_video_size

    for index, row, face_x, face_y, face_w, face_h in zip(
      wipe_df['frame'], wipe_df['video_name_from'],
      wipe_df['FaceRectX'], wipe_df['FaceRectY'], wipe_df['FaceRectWidth'], wipe_df['FaceRectHeight']
    ):
      face_cx = face_x + face_w/2
      face_cy = face_y + face_h/2
      face_sx = int(face_cx - video_w/2)
      face_ex = face_sx + video_w
      face_sy = int(face_cy - video_h/2)
      face_ey = face_sy + video_h
      self.get_interval_frames_cutoff(
        video_name_from=row,
        from_frame=int(index),
        to_frame=int(index+fps),
        video_to=wipe_video,
        clipping=(face_sx, face_ex, face_sy, face_ey),
      )

    wipe_video.release()
    logger.info('Done.')

  def get_interval_frames(self, video_name_from: str, from_frame: int, to_frame: int, video_to) -> None:
    logger.info(f'Frame No.{from_frame} ~ {to_frame-1} from {video_name_from}')
    video_from = cv2.VideoCapture(video_name_from)
    for iframe in range(from_frame, to_frame, 1):
      video_from.set(cv2.CAP_PROP_POS_FRAMES, iframe)
      ret, frame = video_from.read()
      if ret:
        _frame = cv2.resize(frame, self.output_video_size)
        video_to.write(_frame)
    video_from.release()

  def get_interval_frames_cutoff(self, video_name_from: str, from_frame: int, to_frame: int, video_to, clipping: tuple[int, int, int, int]) -> None:
    logger.info(f'Frame No.{from_frame} - {to_frame-1} from {video_name_from}')
    video_from = cv2.VideoCapture(video_name_from)
    (face_sx, face_ex, face_sy, face_ey) = clipping
    for iframe in range(from_frame, to_frame, 1):
      video_from.set(cv2.CAP_PROP_POS_FRAMES, iframe)
      ret, frame = video_from.read()
      if ret:
        _frame = frame[max(0, face_sy):min(frame.shape[1], face_ey), max(0, face_sx):min(frame.shape[0], face_ex)]
        _frame = cv2.resize(_frame, self.output_video_size)
        video_to.write(_frame)
    video_from.release()

  def make_plots(self, column: str, plot_name: str):
    plt.figure()

    for df, video_path in zip(self.dataframes, self.video_paths):
      Xs = df['frame']
      Ys = df[column]
      plt.plot(Xs, Ys, label=os.path.basename(video_path))

    plt.title('1.0: Good, 0.0: Bad')
    plt.xlabel('Frame numbers')
    plt.ylabel('Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_name)

  def make_individual_plots(self, column: str, plot_name: str):
    _, axes = plt.subplots(nrows=len(self.dataframes), ncols=1, figsize=(8, 2 * (len(self.dataframes) + 1)))
    for i, (df, video_path) in enumerate(zip(self.dataframes, self.video_paths)):
      axes[i].plot(df['frame'], df[column], label=os.path.basename(video_path))
      axes[i].set_title(os.path.basename(video_path))
      axes[i].set_xlabel('Frame numbers')
      axes[i].set_ylabel('1: Good, 0: Bad')
      axes[i].grid(True)

    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.savefig(plot_name)

if __name__ == '__main__':
  wiper = Wiper(
    video_paths = [
      '../assets/nico.mp4',
      '../assets/onna.mp4',
      '../assets/otoko.mp4',
      '../assets/sosina.mp4',
    ],
    weights = [
      20.0,
      2.0,
      1.0,
      0.5,
    ],
    interval_in_frame=30,
    csv_paths = [
      '../deliverables/nico.csv',
      '../deliverables/onna.csv',
      '../deliverables/otoko.csv',
      '../deliverables/sosina.csv',
    ],
  )
  wiper.convert_video_to_dataframe()
  wiper.set_dataframes()
  wiper.set_criteria(
    criteria=[
      ('happiness', '> 0.60'),
      ('anger', '< 0.20'),
      ('disgust', '< 0.20'),
      ('fear', '< 0.20'),
      ('sadness', '< 0.20'),
      ('Pitch', '< 30.0'),
      ('Pitch', '> -30.0'),
      ('Roll', '< 30.0'),
      ('Roll', '> -30.0'),
      ('Yaw', '< 30.0'),
      ('Yaw', '> -30.0'),
    ]
  )
  wiper.make_wipe(window_size=10)
