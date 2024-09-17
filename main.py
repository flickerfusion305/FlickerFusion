import argparse
import subprocess

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    # Add arguments
    parser.add_argument('--env', type=str, required=True, help='Specify the environment.')
    parser.add_argument('--method', type=str, required=True, help='Specify the method to use.')

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    env = args.env
    method = args.method

    # Use the arguments in your script
    print(f"Environment: {env}")
    print(f"Method: {method}")

    srckey = {"dotprod": "src_meta_dotprod", "smldg": "src_meta_dotprod", "qmix_atten_daaged": "src_attention", "dgmaml": "src_meta_lr", "mldg": "src_meta_lr",
    "updet": "src_UPDET", "acorm": "src_ACORM", "cama": "src", "qmix_atten_dropout": "src", "qmix_atten": "src", "qmix_dropout": "src",
    "qmix": "src", "refil": "src", "odis_trajectory": "src", "odis_train": "src_ODIS"}

    configkey = {"dotprod": "qmix_atten_DOTPROD", "smldg": "qmix_atten_SMLDG", "qmix_atten_daaged": "qmix_atten", "dgmaml": "qmix_atten_DGMAML", "mldg": "qmix_atten_DGMAML",
    "updet": "qmix_updet", "acorm": "ACORM", "cama": "cama_qmix_atten", "qmix_atten_dropout": "qmix_atten_dropout", "qmix_atten": "qmix_atten", "qmix_dropout": "qmix_dropout",
    "qmix": "qmix", "refil": "refil", "odis_trajectory": "qmix_get_trajectory", "odis_train": "odis"}
    
    envkey = {"adversary": "adversary_mpev2", "guard": "guard_mpev2", "repel": "repel_mpev2", "spread": "spread_mpev2", "tag": "tag_mpev2", "hunt": "hunt_mpev2"}

    src = srckey[method]
    config = configkey[method]
    envconfig = envkey[env]
    
    if src == "src_ACORM":
        command = f"python3 {src}/ACORM_QMIX/main_{env}.py"
    elif src == "src_ODIS":
        command = f"python3 {src}/src/main.py --mto --config={env}/{config} --env-config={envconfig} --task-config={env}-medium"
    else:
        command = f"python3 {src}/main.py --env-config={envconfig} --config={env}/{config}"
    result = subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
