
CREATE DATABASE IF NOT EXISTS disease_prediction
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE disease_prediction;

SHOW TABLES;

-- 1. 단백질 테이블
CREATE TABLE protein (
    uniprot_id VARCHAR(50) NOT NULL,
    gene_id    INT,
    sequence   TEXT,
    PRIMARY KEY (uniprot_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. 질병 테이블
CREATE TABLE disease (
    disease_id VARCHAR(50) NOT NULL,
    name       VARCHAR(255),
    PRIMARY KEY (disease_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. 단백질–질병 매핑 (다대다)
CREATE TABLE protein_disease (
    uniprot_id VARCHAR(50) NOT NULL,
    disease_id VARCHAR(50) NOT NULL,
    PRIMARY KEY (uniprot_id, disease_id),
    FOREIGN KEY (uniprot_id) REFERENCES protein(uniprot_id) ON DELETE CASCADE,
    FOREIGN KEY (disease_id) REFERENCES disease(disease_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. 단백질 상호작용 테이블
CREATE TABLE protein_interaction (
    protein1       VARCHAR(50) NOT NULL,
    protein2       VARCHAR(50) NOT NULL,
    combined_score INT,
    PRIMARY KEY (protein1, protein2),
    FOREIGN KEY (protein1) REFERENCES protein(uniprot_id) ON DELETE CASCADE,
    FOREIGN KEY (protein2) REFERENCES protein(uniprot_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. GO 용어 테이블
CREATE TABLE go_term (
    term_id VARCHAR(50) NOT NULL,
    PRIMARY KEY (term_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. 단백질–GO 매핑 (다대다)
CREATE TABLE protein_go (
    uniprot_id VARCHAR(50) NOT NULL,
    term_id    VARCHAR(50) NOT NULL,
    PRIMARY KEY (uniprot_id, term_id),
    FOREIGN KEY (uniprot_id) REFERENCES protein(uniprot_id) ON DELETE CASCADE,
    FOREIGN KEY (term_id)    REFERENCES go_term(term_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7. PDB 구조 테이블
CREATE TABLE pdb_structure (
    pdb_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (pdb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 8. 단백질–PDB 매핑 (다대다)
CREATE TABLE protein_pdb (
    uniprot_id VARCHAR(50) NOT NULL,
    pdb_id     VARCHAR(20) NOT NULL,
    PRIMARY KEY (uniprot_id, pdb_id),
    FOREIGN KEY (uniprot_id) REFERENCES protein(uniprot_id) ON DELETE CASCADE,
    FOREIGN KEY (pdb_id)     REFERENCES pdb_structure(pdb_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 9. 단백질 별칭 테이블
CREATE TABLE protein_alias (
    alias_id   INT          NOT NULL AUTO_INCREMENT,
    uniprot_id VARCHAR(50)  NOT NULL,
    alias      TEXT,
    PRIMARY KEY (alias_id),
    FOREIGN KEY (uniprot_id) REFERENCES protein(uniprot_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

