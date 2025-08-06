
CREATE DATABASE IF NOT EXISTS disease_prediction
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE disease_prediction;

-- 1. 단백질 테이블
CREATE TABLE protein (
    sequence_id VARCHAR(100) NOT NULL,
    sequence    TEXT,
    gene_id     INT,
    PRIMARY KEY (sequence_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO protein (sequence_id, sequence, gene_id)
VALUES ('SEQ001', 'MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQK', 1234);

-- 2. 질병 테이블
CREATE TABLE disease (
    disease_id VARCHAR(50) NOT NULL,
    name       VARCHAR(255),
    PRIMARY KEY (disease_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO disease (disease_id, name)
VALUES ('D001', 'Alzheimer Disease');

-- 3. 단백질–질병 매핑 (다대다)
CREATE TABLE protein_disease (
    sequence_id VARCHAR(100) NOT NULL,
    disease_id  VARCHAR(50) NOT NULL,
    PRIMARY KEY (sequence_id, disease_id),
    FOREIGN KEY (sequence_id) REFERENCES protein(sequence_id) ON DELETE CASCADE,
    FOREIGN KEY (disease_id) REFERENCES disease(disease_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO protein_disease (sequence_id, disease_id)
VALUES ('SEQ001', 'D001');

-- 4. 단백질 상호작용 테이블
CREATE TABLE protein_interaction (
    sequence_id_1 VARCHAR(100) NOT NULL,
    sequence_id_2 VARCHAR(100) NOT NULL,
    combined_score INT,
    PRIMARY KEY (sequence_id_1, sequence_id_2),
    FOREIGN KEY (sequence_id_1) REFERENCES protein(sequence_id) ON DELETE CASCADE,
    FOREIGN KEY (sequence_id_2) REFERENCES protein(sequence_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO protein_interaction (sequence_id_1, sequence_id_2, combined_score)
VALUES ('SEQ001', 'SEQ001', 980);

-- 5. GO 용어 테이블
CREATE TABLE go_term (
    term_id VARCHAR(50) NOT NULL,
    PRIMARY KEY (term_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO go_term (term_id)
VALUES ('GO:0003674');

-- 6. 단백질–GO 매핑 (다대다)
CREATE TABLE protein_go (
    sequence_id VARCHAR(100) NOT NULL,
    term_id     VARCHAR(50) NOT NULL,
    PRIMARY KEY (sequence_id, term_id),
    FOREIGN KEY (sequence_id) REFERENCES protein(sequence_id) ON DELETE CASCADE,
    FOREIGN KEY (term_id) REFERENCES go_term(term_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO protein_go (sequence_id, term_id)
VALUES ('SEQ001', 'GO:0003674');

-- 7. PDB 구조 테이블
CREATE TABLE pdb_structure (
    pdb_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (pdb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO pdb_structure (pdb_id)
VALUES ('1ABC');

-- 8. 단백질–PDB 매핑 (다대다)
CREATE TABLE protein_pdb (
    sequence_id VARCHAR(100) NOT NULL,
    pdb_id      VARCHAR(20) NOT NULL,
    PRIMARY KEY (sequence_id, pdb_id),
    FOREIGN KEY (sequence_id) REFERENCES protein(sequence_id) ON DELETE CASCADE,
    FOREIGN KEY (pdb_id) REFERENCES pdb_structure(pdb_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO protein_pdb (sequence_id, pdb_id)
VALUES ('SEQ001', '1ABC');

-- 9. 단백질 별칭 테이블
CREATE TABLE protein_alias (
    alias_id    INT NOT NULL AUTO_INCREMENT,
    sequence_id VARCHAR(100) NOT NULL,
    alias       TEXT,
    PRIMARY KEY (alias_id),
    FOREIGN KEY (sequence_id) REFERENCES protein(sequence_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO protein_alias (sequence_id, alias)
VALUES ('SEQ001', 'Amyloid beta precursor protein');

-- 10. 예측 결과 테이블
CREATE TABLE prediction_result (
    prediction_id        INT NOT NULL AUTO_INCREMENT,
    sequence             TEXT,
    predicted_disease_id VARCHAR(50),
    confidence_score     FLOAT,
    predicted_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prediction_id),
    FOREIGN KEY (predicted_disease_id) REFERENCES disease(disease_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 예시 insert
INSERT INTO prediction_result (sequence, predicted_disease_id, confidence_score)
VALUES ('MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQK', 'D001', 0.921);

-- 테이블 목록 확인
SHOW TABLES;