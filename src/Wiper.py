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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Handmade modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FaceExpressionWithPyFeat.src.facial_expression import FacialExpressionAnalysis

def clip(image_name: str, top: int, bottom: int, left: int, right: int):
  image = cv2.imread(image_name)
  image = image[top:bottom, left:right]
  cv2.imwrite('output.png', image)

class Wiper(object):
  def __init__(
      self,
      video_paths: list[str], # mp4などの動画ファイルのパス
      weights: list[float], # どれだけ対応する動画ファイルを重要視するか
      interval_in_frame: int=30, # 表情検出、角度検出などを何フレームおきに実行するか、秒じゃないよ
      csv_paths: Union[list[str], None]=None, # 前回結果を保存したCSVファイルがある場合は指定して時短
      criteria: Union[list[tuple[str, str]], None]=None, # どのような条件でワイプ動画を選定するか
      scheduled_in_frames: list[list[tuple[int, int]]]=None, # 前もって決定されているワイプ(喋っているの場所など)
      output_video_name: str='output.mp4', # 出力ビデオの名前
      output_video_size: tuple[int, int]=(800, 800), # 出力ビデオのサイズ in フレームサイズ
  ) -> None:
    self.video_paths = video_paths
    self.weights = weights
    self.interval_in_frame = interval_in_frame
    self.csv_paths = [] if csv_paths is None else csv_paths
    self.dataframes = []
    self.criteria = criteria
    self.scheduled_in_frames = scheduled_in_frames
    self.output_video_name = output_video_name
    self.output_video_size = output_video_size

  def convert_videos_to_dataframe(self) -> None:
    # 動画ファイルをpandas.DataFrameに変換する
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
    # データフレームをCSVファイルから取得
    parser = FacialExpressionAnalysis()
    return parser.read_results(csvfile=csv_path)

  def set_dataframes(self) -> None:
    # データフレームを属性値に
    for csv_path in self.csv_paths:
      dataframe = self.get_dataframe(csv_path=csv_path)
      self.dataframes.append(dataframe)

  def set_criteria(self, criteria: list[tuple[str, str]]) -> None:
    # 条件を設定
    self.criteria = criteria

  def set_scheduled_in_frames(self, scheduled_in_frames: list[list[tuple[int, int]]]) -> None:
    # 前もって決定されているワイプの設定
    '''
    入力のscheduleは以下の形式を想定
    scheduled_in_frames = [
      [(), ()], # 1つ目のビデオに対するワイプ
      [(), ()], # 2つ目のビデオに対するワイプ
      ...
    ]
    '''
    self.scheduled_in_frames = scheduled_in_frames

  def get_criteria_str(self) -> str:
    '''
    入力のcriteriaは以下の形式を想定
      criteria = [
        ('happiness', '> 0.9'),
        ('Pitch', '< 15.0'),
        ('Pitch', '> -15.0'),
      ]
    '''
    criteria_str =' and '.join([f'{criterion[0]} {criterion[1]}' for criterion in self.criteria])
    return criteria_str

  def get_scheduled_in_record_id(self) -> None:
    # 前もって決定されているワイプの取得
    scheduled_in_record_id = []
    for scheduled_in_frames in self.scheduled_in_frames:
      scheduled_in_record_id.append([(int(from_frame / self.interval_in_frame), int(to_frame / self.interval_in_frame)) for (from_frame, to_frame) in scheduled_in_frames])
    return scheduled_in_record_id

  def get_fulfil_records(self, df: pd.DataFrame) -> pd.DataFrame:
    # 条件に合うレコードを取得
    criteria = self.get_criteria_str()
    return df.query(criteria)

  def add_columns_with_criteria(self, df: pd.DataFrame, column_name: str, true_val: Any, false_val: Any):
    # 新しいカラムを追加(criteriaを満たす -> true_val, 満たさない -> false_val)
    df[column_name] = True
    for criterion in self.criteria:
      df[column_name] = [true_val if eval(f'{x} {criterion[1]} and {y}') else false_val for x, y in zip(df[criterion[0]], df[column_name])]

  def add_boolean_columns_with_criteria(self, df: pd.DataFrame, column_name: str):
    # 新しいカラムを追加(criteriaを満たす -> True, 満たさない -> False)
    self.add_columns_with_criteria(df=df, column_name=column_name, true_val=True, false_val=False)

  def add_bit_columns_with_criteria(self, df: pd.DataFrame, column_name: str):
    # 新しいカラムを追加(criteriaを満たす -> 1, 満たさない -> 0)
    self.add_columns_with_criteria(df=df, column_name=column_name, true_val=1, false_val=0)

  def add_float_columns_with_criteria(self, df: pd.DataFrame, column_name: str, true_val: float, false_val: float):
    # 新しいカラムを追加(criteriaを満たす -> true_val: float, 満たさない -> falce_val: float)
    self.add_columns_with_criteria(df=df, column_name=column_name, true_val=true_val, false_val=false_val)

  def add_columns_with_moving_sum(self, df: pd.DataFrame, window_size: int, column: str):
    # window_sizeの移動和を取る
    df[f'{column}_move_sum_{window_size}'] = df[column].rolling(window=window_size, min_periods=1).sum().shift(-window_size+1)
    df[f'{column}_move_sum_{window_size}'] = df[f'{column}_move_sum_{window_size}'].fillna(0)

  def add_columns_with_moving_mean(self, df: pd.DataFrame, window_size: int, column: str):
    # window_sizeの移動平均を取る
    df[f'{column}_move_mean_{window_size}'] = df[column].rolling(window=window_size, min_periods=1).mean().shift(-window_size+1)
    df[f'{column}_move_mean_{window_size}'] = df[f'{column}_move_mean_{window_size}'].fillna(0)

  def make_wipe(self, window_size: int=10):
    # ワイプの作成
    column = 'fulfil'
    for df, video_path, weight, schedules in zip(self.dataframes, self.video_paths, self.weights, self.get_scheduled_in_record_id()):
      # はじめにどのレコードが条件を満たすかをbit値で表現する
      self.add_float_columns_with_criteria(df=df, column_name=column, true_val=weight, false_val=0.0)

      # 各レコードに対し、window_sizeの移動和をとる
      self.add_columns_with_moving_sum(df=df, window_size=window_size, column=column)

      # わかりやすいように列名を変えておく
      df[video_path] = df[f'{column}_move_sum_{window_size}']
      for facecolumn in ['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']:
        df[f'{video_path}_{facecolumn}'] = df[facecolumn]

      # 前もってワイプに設定されているフレームに関する操作
      for (from_frame, to_frame) in schedules:
        if from_frame <= to_frame-1:
          df.loc[from_frame:to_frame-1, video_path] = self.interval_in_frame * max(self.weights) * 2

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
          how='outer',
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
            how='outer',
          )

    merged_df.to_csv('merged_df.csv')

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

    # 出力ビデオの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    wipe_video = cv2.VideoWriter(self.output_video_name, fourcc, self.interval_in_frame, self.output_video_size)
    video_width, video_height = self.output_video_size

    recorded_latest_frame_no = -1
    for frame, video_name_from, left, top, width, height in zip(
      wipe_df['frame'], wipe_df['video_name_from'],
      wipe_df['FaceRectX'], wipe_df['FaceRectY'], wipe_df['FaceRectWidth'], wipe_df['FaceRectHeight']
    ):
      if frame <= recorded_latest_frame_no:
        continue
      elif frame == recorded_latest_frame_no + 1:
        face_left   = int(left + width/2  - video_width/2)
        face_right  = face_left + video_width
        face_top    = int(top  + height/2 - video_height/2)
        face_bottom = face_top + video_height

        self.get_interval_frames_cutoff(
          video_name_from=video_name_from,
          from_frame=int(frame),
          to_frame=int(frame+self.interval_in_frame),
          video_to=wipe_video,
          clipping=(face_top, face_bottom, face_left, face_right),
        )
      else:
        logger.info(f'{self.interval_in_frame} black frames are inserted.')
        for _ in range(self.interval_in_frame):
          black = np.zeros((video_width, video_height, 3), dtype = np.uint8)
          wipe_video.write(black)

      recorded_latest_frame_no = frame+self.interval_in_frame-1

    wipe_video.release()
    logger.info('Done.')

  def get_interval_frames(self, video_name_from: str, from_frame: int, to_frame: int, video_to) -> None:
    # from_frameからto_frameのフレームを取得
    logger.info(f'Frame No.{from_frame} ~ {to_frame-1} from {video_name_from}')
    video_from = cv2.VideoCapture(video_name_from)
    for iframe in range(from_frame, to_frame, 1):
      video_from.set(cv2.CAP_PROP_POS_FRAMES, iframe)
      ret, frame = video_from.read()
      if ret:
        # clipping -> image[top:bottom, left:right]
        _frame = cv2.resize(frame, self.output_video_size)
        video_to.write(_frame)
    video_from.release()

  def get_interval_frames_cutoff(self, video_name_from: str, from_frame: int, to_frame: int, video_to, clipping: tuple[int, int, int, int]) -> None:
    # from_frameからto_frameのフレームを整形し取得
    logger.info(f'Frame No.{from_frame} - {to_frame-1} from {video_name_from}')
    video_from = cv2.VideoCapture(video_name_from)
    (top, bottom, left, right) = clipping
    for iframe in range(from_frame, to_frame, 1):
      video_from.set(cv2.CAP_PROP_POS_FRAMES, iframe)
      ret, frame = video_from.read()
      if ret:
        # clipping -> image[top:bottom, left:right]
        _frame = frame[max(0, top):min(frame.shape[0], bottom), max(0, left):min(frame.shape[1], right)]
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
      '../assets/BlackNerdComedy_CapCut.mp4',
      '../assets/DeanBarry_CapCut.mp4',
      '../assets/funnylilgalreacts_CapCut.mp4',
    ],
    weights = [
      1.0,
      2.0,
      1.5,
    ],
    interval_in_frame=30,
    csv_paths = [
      '../deliverables/BlackNerdComedy_CapCut.csv',
      '../deliverables/DeanBarry_CapCut.csv',
      '../deliverables/funnylilgalreacts_CapCut.csv',
    ],
    output_video_name='pool.mp4',
  )
  # wiper.convert_videos_to_dataframe()
  wiper.set_dataframes()
  wiper.set_criteria(
    criteria=[
      #('happiness', '> 0.60'),
      #('anger', '< 0.20'),
      #('disgust', '< 0.20'),
      #('fear', '< 0.20'),
      #('sadness', '< 0.20'),
      ('Pitch', '< 60.0'),
      ('Pitch', '> -60.0'),
      ('Roll', '< 60.0'),
      ('Roll', '> -60.0'),
      ('Yaw', '< 60.0'),
      ('Yaw', '> -60.0'),
    ]
  )
  wiper.set_scheduled_in_frames(
    scheduled_in_frames=[
      [(300, 390), (870, 930)],
      [(0, 150), (450, 510)],
      [((660, 720))],
    ]
  )
  wiper.make_wipe(window_size=10)