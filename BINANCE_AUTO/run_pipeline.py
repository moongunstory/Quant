#  python C:\gtpbitcoin\BINANCE_AUTO\run_pipeline.py --lgbm / lgbm만 실행하는 명령어













import os
import sys
import subprocess
import argparse

def main():
    """
    모방 학습 후 디버그 가치 사전 학습을 순차적으로 실행하는 파이프라인 스크립트.
    명령어 인자로 원하는 스크립트만 선택 실행 가능.
    """
    
    # 명령어 인자 파서 설정
    parser = argparse.ArgumentParser(description="ML 파이프라인 선택적 실행")
    parser.add_argument('--processor', action='store_true', help='데이터 전처리만 실행')
    parser.add_argument('--lgbm', action='store_true', help='LGBM 학습만 실행')
    parser.add_argument('--imitation', action='store_true', help='모방 학습만 실행')
    parser.add_argument('--debug-value', action='store_true', help='디버그 가치 사전 학습만 실행')
    
    args = parser.parse_args()
    
    # 스크립트가 위치한 디렉토리를 기준으로 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 실행할 스크립트 경로 정의
    processor_script = os.path.join(base_dir, 'modules', 'training', 'data_preparation', 'processor.py')
    lgbm_script = os.path.join(base_dir, 'modules', 'training', 'lgbm', 'train_lgbm.py')
    imitation_script = os.path.join(base_dir, 'modules', 'training', 'ppo', 'imitation', 'train_imitation.py')
    debug_value_script = os.path.join(base_dir, 'modules', 'training', 'ppo', 'debug_value_pretrain.py')
    
    # 전체 스크립트 목록
    all_scripts = [
        ("데이터 전처리 (Processor)", processor_script, "processor_output.txt"),
        ("LGBM 학습", lgbm_script, "lgbm_training_output.txt"),
        ("모방 학습", imitation_script, "imitation_training_output.txt"),
        ("디버그 가치 사전 학습", debug_value_script, None) # 디버그 스크립트는 결과 저장 안함
    ]
    
    # 실행할 스크립트 선택
    scripts_to_run = []
    
    # 명령어 인자에 따라 스크립트 선택
    if args.processor:
        scripts_to_run.append(all_scripts[0])
    if args.lgbm:
        scripts_to_run.append(all_scripts[1])
    if args.imitation:
        scripts_to_run.append(all_scripts[2])
    if args.debug_value:
        scripts_to_run.append(all_scripts[3])
    
    # 아무 인자도 없으면 전체 실행 (기존 동작)
    if not any([args.processor, args.lgbm, args.imitation, args.debug_value]):
        scripts_to_run = all_scripts
        print(">>> 파이프라인 시작: 모방 학습 -> 디버그 가치 사전 학습")
    else:
        print(f">>> 선택된 스크립트 실행: {[name for name, _, _ in scripts_to_run]}")
    
    for name, script_path, output_filename in scripts_to_run:
        print(f"\n{'='*20} [{name}] 실행 시작 {'='*20}")
        
        try:
            # check=True: 스크립트 실행 중 오류 발생 시 예외를 발생시킴
            # capture_output=True: stdout과 stderr를 캡처
            # text=True: 캡처된 출력을 텍스트로 디코딩
            result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
            print(f"✅ [{name}] 실행 완료")
            
            if output_filename:
                output_path = os.path.join(base_dir, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"--- {name} Standard Output ---\n")
                    f.write(result.stdout)
                    f.write(f"\n--- {name} Standard Error ---\n")
                    f.write(result.stderr)
                print(f"   ➡️ [{name}] 실행 결과 저장: {output_path}")
        
        except subprocess.CalledProcessError as e:
            print(f"❌ [{name}] 실행 중 오류 발생! 파이프라인을 중단합니다.")
            print(f"오류 코드: {e.returncode}")
            print(f"Standard Output:\n{e.stdout}")
            print(f"Standard Error:\n{e.stderr}")
            # 오류 발생 시 전체 스크립트 중단
            sys.exit(1)
        
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {script_path}")
            sys.exit(1)
    
    print(f"\n{'='*20} 모든 파이프라인 작업 완료 {'='*20}")

if __name__ == "__main__":
    main()