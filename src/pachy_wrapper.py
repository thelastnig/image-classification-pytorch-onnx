import subprocess

pachctl = 'pachctl'
ctx_name = 'trainer'

try:
  subprocess.run(pachctl)
except FileNotFoundError:
  raise EnvironmentError(f"{pachctl} is not installed")


def setup(pachy_host, pachy_port):
  try:
    subprocess.run(f"{pachctl} config update"
                   f" --pachd-address {pachy_host}:{pachy_port}")
  except:
    raise EnvironmentError(f"Failed to connect to pachyderm cluster at"
                           f" {pachy_host}:{pachy_port}")


def get_file(commit, branch, path, recursive=False):
  try:
    subprocess.run(f"{pachctl} get file {'-r ' if recursive else ''}"
                   f"{commit}@{branch}:{path}")
  except:
    raise EnvironmentError(f"Failed to get file from pachyderm cluster")
