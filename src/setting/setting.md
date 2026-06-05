# 윈도우 CMD 창 오픈

```bash
wsl --install     # 우분투 설치

# wsl 실행 후 우분투 내부에서
cat /etc/os-release    # 우분투 설치 버전 확인
```


# 설치 가능한 배포판 목록 확인

```bash
wsl --list --online
```


# 특정 버전 설치 (예: 24.04)

```bash
wsl --install -d Ubuntu-24.04
```


# 파이썬 설치

```bash
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.10 python3.10-venv python3.10-dev

python3.10 --version
```

---

# wsl 실행

```bash
cd ~     # 홈 디렉토리로 이동
```

## 1. 가상환경 생성 및 활성화

```bash
python3.10 -m venv tf_env
source tf_env/bin/activate
```

## 2. 최신 TensorFlow 및 CUDA 패키지 설치

```bash
pip install --upgrade pip
pip install tensorflow[and-cuda]
```

## 3. GPU 인식 확인

```bash
python -c "import tensorflow as tf; print('\n' + '='*50); print('사용 가능한 GPU:', tf.config.list_physical_devices('GPU')); print('='*50)"
```

## 4. cuDNN / cuBLAS 심볼릭 링크 생성

```bash
# cuDNN: .so.9 → .so.8 링크 생성
cd $VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib && \
for file in *.so.9; do ln -sf "$file" "${file%.9}.8"; done && \
cd ~

# cuBLAS: .so.12 → .so.11 링크 생성
cd $VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib && \
for file in *.so.12; do ln -sf "$file" "${file%.12}.11"; done && \
cd ~
```

## 5. LD_LIBRARY_PATH 설정

```bash
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/curand/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/nccl/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/nvtx/lib:$LD_LIBRARY_PATH
```

## 6. 동작 확인

```bash
python -c "import tensorflow as tf; print('연산 테스트:', tf.matmul([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]))"
```


# 설정을 영구적으로 저장

```bash
nano tf_env/bin/activate
```

파일 맨 아래에 아래 내용 복사/붙여넣기:

```bash
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/curand/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/nccl/lib:$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/nvtx/lib:$LD_LIBRARY_PATH
```

`Ctrl + O`, `Enter`, `Ctrl + X`로 저장하고 나옵니다.


# 새로운 터미널을 열었을 때

```bash
source tf_env/bin/activate
```

동작 확인:

```bash
python -c "import tensorflow as tf; print('연산 테스트:', tf.matmul([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]))"
```


# 파이참 인터프리터 환경 세팅

1. 우분투에서 생성한 가상환경 디렉토리 선택해서 오픈
2. 인터프리터 추가
   - 설정 → 인터프리터 → 인터프리터 추가 → WSL 선택
   - → 기존 항목 선택 → Python 경로 선택
   - 예) WSL (Ubuntu): (/home/sckit/tf_env/bin/python3.10)

파이참 터미널:

```bash
cd ~
source tf_env/bin/activate

echo $LD_LIBRARY_PATH
```

출력 예시:

```
/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/curand/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/nccl/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/nvtx/lib:
```

출력 내용을 구성 편집(Run Configuration) 메뉴에서 환경 변수로 등록:

1. **파이참 환경 변수 창 열기**: 해당 실행 구성(Run Configuration)의 환경 변수 입력창 오른쪽 끝에 있는 문서 모양 아이콘을 클릭하여 전용 창을 띄웁니다.
2. **항목 수정 또는 추가**:
   - 이미 `LD_LIBRARY_PATH`라는 이름이 목록에 있다면, 해당 행의 '값(Value)' 칸을 더블클릭합니다.
   - 기존에 적혀 있던 내용을 지우거나, 그 뒤에 콜론(`:`)을 입력한 후 복사한 내용을 붙여넣습니다.
   - 만약 목록에 없다면 좌측 상단의 `+` 버튼을 눌러 이름을 `LD_LIBRARY_PATH`로 만들고 값을 붙여넣습니다.
3. **확인 및 적용**: 하단의 확인(OK) 버튼을 눌러 창을 닫고, 실행 구성 창에서도 적용(Apply) 또는 확인을 누릅니다.

등록할 환경 변수:

- 이름: `LD_LIBRARY_PATH`
- 값:

```
/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/curand/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/nccl/lib:/home/sckit/tf_env/lib/python3.10/site-packages/nvidia/nvtx/lib:
```

> 파일 단위가 아닌 **프로젝트 단위 적용 시**
> → 파이참_CUDA_Path설정 디렉토리 내부 매뉴얼 참조


# 버전 호환 패키지

```bash
# TensorFlow 2.21 버전 호환 matplotlib
pip install "matplotlib>=3.10.0"

# TensorFlow 2.21 버전 호환 scikit-learn
pip install "scikit-learn>=1.7.0,<1.8.0"
```


# 추가 팁: 원활한 학습을 위해

학습 중에 `InternalError: CUDA_ERROR_OUT_OF_MEMORY`가 발생한다면, RTX 5080의 VRAM을 TensorFlow가 시작부터 100% 다 점유하지 않도록 코드 상단에 아래 설정을 넣어주는 것이 좋습니다.

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

---

# VS Code + WSL 개발 환경 세팅

## 1단계: VS Code 설치하기 (Windows)

1. **공식 웹사이트 접속**: 브라우저를 열고 [VS Code 공식 홈페이지](https://code.visualstudio.com/)로 이동합니다.
2. **다운로드**: 화면의 **[Download for Windows]** 버튼을 눌러 설치 파일을 받습니다.
3. **설치 진행**: 다운로드한 `.exe` 파일을 실행합니다.

> **꿀팁**: 설치 중 '추가 작업 선택' 창이 나오면, **Code로 열기(Open with Code)**를 Windows 탐색기 메뉴에 추가하는 항목들을 모두 체크해 주세요. 나중에 폴더를 열 때 매우 편리합니다. (그냥 모두 체크)


## 2단계: WSL 및 Linux(Ubuntu) 설치 확인

> 이미 WSL과 Ubuntu가 설치되어 있다면 3단계로 넘어가도 됩니다.

1. 시작 메뉴에서 **PowerShell**을 **관리자 권한**으로 실행합니다.
2. 다음 명령어로 WSL과 기본 리눅스(Ubuntu)를 설치합니다.

   ```powershell
   wsl --install
   ```

3. 설치가 완료되면 컴퓨터를 **재부팅**합니다.
4. 재부팅 후 자동으로 뜨는 우분투 터미널에서 리눅스용 **Username(ID)**과 **Password**를 설정합니다. (기억하기 쉬운 값으로 설정)


## 3단계: VS Code에 'WSL 확장 프로그램' 설치하기

윈도우의 VS Code가 리눅스(WSL) 내부를 들여다볼 수 있도록 다리를 놓아주는 작업입니다.

1. 설치한 VS Code를 실행합니다.
2. 왼쪽 사이드바에서 블록 모양의 **Extensions(확장)** 탭을 클릭합니다. (단축키: `Ctrl + Shift + X`)
3. 검색창에 **WSL**을 검색합니다.
4. Microsoft에서 만든 **WSL** 확장(예전 이름: Remote - WSL)을 찾아 **[Install]** 버튼을 누릅니다.


## 4단계: VS Code를 WSL 환경으로 접속하기

1. VS Code 왼쪽 **최하단 구석**의 파란색(또는 초록색) 원격 연결 아이콘 `><` 을 클릭합니다.
2. 상단에 뜨는 메뉴에서 **[Connect to WSL]** (또는 **[New WSL Window]**)을 선택합니다.

> **성공 확인법**: VS Code 왼쪽 아래에 **WSL: Ubuntu** 문구가 표시되면 리눅스 원격 접속 성공입니다!


## 5단계: WSL 전용 파이썬 인터프리터 연동하기 (Python 기준)

리눅스 내부의 파이썬을 인식시키는 최종 단계입니다.

1. WSL이 연결된 VS Code 창에서 다시 **Extensions(확장)** 탭(`Ctrl + Shift + X`)을 엽니다.
2. **Python**을 검색하여 Microsoft에서 만든 **Python** 확장을 설치합니다.
3. 왼쪽 **탐색기** 버튼을 클릭 → **Open Folder** 선택
4. `/home/본인/가상환경폴더`를 선택하고 **OK**.
   - 예) `/home/sckit/tf_env`
5. 왼쪽 맨 아래 **톱니바퀴(⚙) → 설정(Settings) → Text Editor**에서 폰트를 설정합니다.
6. 탐색기 창의 빈 곳에서 **마우스 오른쪽 클릭 → 새 폴더 생성**, 새 폴더 안에서 **새 파일 생성**.
