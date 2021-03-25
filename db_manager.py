import os
import pymysql
import datetime

DELETED_STATUS = 'Deleted_Internal'
ERROR_ID = 'EDP-009'

class DBManager:
    def __init__(self):
        self.SM_AIP_DB_NAME = os.environ['SM_AIP_DB_NAME']
        self.ML_FLOW_DB_NAME = os.environ['ML_FLOW_DB_NAME']
        self.SM_AIP_DB_USER = os.environ['SM_AIP_DB_USER']
        self.SM_AIP_DB_PASSWORD = os.environ['SM_AIP_DB_PASSWORD']
        self.SM_AIP_DB_HOST = os.environ['SM_AIP_DB_HOST']
        self.SM_AIP_DB_PORT = os.environ['SM_AIP_DB_PORT']

        self.connection = pymysql.connect(host=self.SM_AIP_DB_HOST,
                                          user=self.SM_AIP_DB_USER,
                                          password=self.SM_AIP_DB_PASSWORD)

    # 최대 버전 수 초과 여부 검사
    def check_exceed_version_limit(self, name):
        try:
            with self.connection.cursor() as cursor:
                sql_for_auto_delete_table = f"SELECT max_num FROM {self.SM_AIP_DB_NAME}.TB_AUTO_DELETE WHERE name = %s"
                cursor.execute(sql_for_auto_delete_table, (name,))
                max_num_from_auto_delete_table = cursor.fetchone()[0]

                if max_num_from_auto_delete_table == 0:
                    return False

                sql_for_model_versions_table = f"""
                    SELECT COUNT(*)
                    FROM {self.ML_FLOW_DB_NAME}.model_versions
                    WHERE name = %s AND current_stage != %s
                """
                cursor.execute(sql_for_model_versions_table, (name, DELETED_STATUS))
                max_num_from_model_versions_table = cursor.fetchone()[0]

                return max_num_from_model_versions_table >= max_num_from_auto_delete_table

        finally:
            self.connection.close()

    # # 공통. 최대 버전 수 이상일 경우 가장 오래된 버전 삭제
    # def delete_oldest_version(self, name):
    #     try:
    #         with connection.cursor() as cursor:
    #             sql = f"""
    #                 SELECT MIN(version),
    #                 run_id
    #                 FROM {self.ML_FLOW_DB_NAME}.model_versions
    #                 WHERE name = %s AND CURRENT_STAGE != %s
    #             """
    #             cursor.execute(sql, (name, DELETED_STATUS))
    #             version = cursor.fetchone()[0]
    #             run_id = cursor.fetchone()[1]
    #             try:
    #                 experiment = ExperimentDao.objects.get(run_uuid=run_id)
    #             except ObjectDoesNotExist:
    #                 raise ExpeirmentNotExistException()
    #             mlflowManager.delete_model_version(name, str(version))
    #             ExperimentDao.objects.set_item(experiment.experiment_id, {"working_status": SUCCEEDED})
    #     finally:
    #         connection.close()

    # 버전 생성 시 필요 정보 등록
    def set_model_version(self, name, version, user_id):
        try:
            with self.connection.cursor() as cursor:
                sql = f"""
                    SELECT MAX(version)
                    FROM {self.ML_FLOW_DB_NAME}.model_versions
                    WHERE name=%s
                """
                cursor.execute(sql, (name,))
                last_version = cursor.fetchone()[0]

                status_message = f"onnx:{version}"

                update_sql = f"""
                    UPDATE {self.ML_FLOW_DB_NAME}.model_versions 
                    SET user_id=%s,
                    status_message=%s
                    WHERE name=%s AND version=%s
                """
                cursor.execute(update_sql, (user_id, status_message, name, last_version))

                update_status_sql = f"""
                    UPDATE {self.ML_FLOW_DB_NAME}.model_versions 
                    SET current_stage=%s
                    WHERE name=%s AND version=%s
                """
                cursor.execute(update_status_sql, ('None', name, version))

                previous_log_message = f"ONNX conversion completed"
                next_log_message = f"ONNX conversion from {version} version"
                sql = f"""
                    INSERT INTO {self.SM_AIP_DB_NAME}.TB_MODEL_VERSION_LOG
                    (name, version, log, create_date, update_date, create_user, update_user)
                    VALUES (%s, %s, %s, now(), now(), %s, %s)
                """
                cursor.execute(sql, (name, version, previous_log_message, user_id, user_id))
                cursor.execute(sql, (name, last_version, next_log_message, user_id, user_id))
                self.connection.commit()
        finally:
            self.connection.close()

    # 포맷 변환 실패 시 알람 등록
    def set_fail_alarm(self, user_id, log, project_id):
        try:
            with self.connection.cursor() as cursor:
                sql = f"""
                    INSERT INTO {self.SM_AIP_DB_NAME}.TB_ERROR_LOG
                    (error_id, project_id, log_detail, create_user, create_date, read_yn)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (ERROR_ID, project_id, log, user_id, datetime.datetime.now(), False))

                self.connection.commit()
        finally:
            self.connection.close()