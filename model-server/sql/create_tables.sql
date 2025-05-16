CREATE DATABASE IF NOT EXISTS disease_prediction
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE disease_prediction;

-- 1. 단백질 테이블 (protein)
CREATE TABLE protein (
    uniprot_id     VARCHAR(100) NOT NULL,
    sequence       TEXT NOT NULL,
    PRIMARY KEY (uniprot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. 질병 테이블 (disease)
CREATE TABLE disease (
    disease_id     VARCHAR(100) NOT NULL,
    disease_name   VARCHAR(100) NOT NULL,
    PRIMARY KEY (disease_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. 단백질-질병 매핑 테이블 (protein-disease)
CREATE TABLE protein_disease (
    uniprot_id     VARCHAR(100) NOT NULL,
    disease_id     VARCHAR(100) NOT NULL,
    PRIMARY KEY (uniprot_id, disease_id),
    FOREIGN KEY (uniprot_id) REFERENCES protein(uniprot_id),
    FOREIGN KEY (disease_id) REFERENCES disease(disease_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. 예측된 단백질 테이블 (predict_protein)
CREATE TABLE predict_protein (
    run_id         INT NOT NULL,
    `rank`         INT NOT NULL,
    sequence       TEXT NOT NULL,
    PRIMARY KEY (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. 모델 예측 로그 테이블 (model_prediction_run)
CREATE TABLE model_prediction_run (
    run_id         INT NOT NULL,
    input_sequence TEXT NOT NULL,
    created_at     DATE NOT NULL,
    model_version  VARCHAR(20) NOT NULL,
    PRIMARY KEY (run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. 예측된 질병 테이블 (predict_disease)
CREATE TABLE predict_disease (
    run_id         INT NOT NULL,
    disease_name   VARCHAR(100) NOT NULL,
    `rank`         INT NOT NULL,
    PRIMARY KEY (run_id),
    FOREIGN KEY (run_id) REFERENCES predict_protein(run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;



-- 테이블 목록 확인
SHOW TABLES;