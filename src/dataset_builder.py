"""
Constructor, limpiador y versionador del conjunto de datos de retroalimentación.

Implementa:
- [Parte 1.1 — Registro de versión del conjunto de datos]: genera un manifiesto
  con versión semántica, huella SHA-256 del CSV limpio, modo de origen, conteos y
  distribución de clases.
- Soporte dual transparente:
  1. **Modo `real_csv`**: si los tres archivos CSV del trabajo parcial
     (`feedback_curso_universitario.csv`, `feedback_curso_universitario2.csv`,
     `feedback_curso_universitario3.csv`) están presentes en `data/raw/` o en la
     raíz del proyecto, replica con exactitud el saneamiento del Paso 1 y Paso 2
     de `Parcial_SI_corregido.ipynb` (eliminación de cabeceras `id == 'id'`,
     corrección de `Sistemas opertativos`, canonización de `polaridad_basica`,
     parseo de `aspectos` y correcciones auditadas por LLM usando la llave
     compuesta `(origen_archivo, id)`).
  2. **Modo `synthetic_parcial_replica`**: si los CSV originales no se encuentran
     aún en el directorio local, construye un dataset reproducible de 618 registros
     en español con las mismas propiedades estadísticas y lingüísticas reportadas
     en `Informe_Parcial_SisInt.pdf` (distribución 258 neutral / 205 negativo /
     155 positivo, 7 asignaturas, 4 semestres, ~76% cláusulas contrastantes, casos
     auditados de la Tabla 1 y etiquetas estructuradas de los 9 aspectos).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    COURSES_CANONICAL,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_CSV_FILENAMES,
    RAW_DATA_DIR,
    SEMESTERS_CANONICAL,
)
from src.preprocessing import (
    corregir_polaridad,
    detectar_contraste,
    normalizar_texto,
    parsear_aspectos,
)

# Correcciones auditadas en el Paso 2 de `Parcial_SI_corregido.ipynb` (Tabla 1 del informe)
CORRECCIONES_LLM_ARCHIVO_1: Dict[int, str] = {
    3: "positivo",
    6: "positivo",
    9: "negativo",
    11: "negativo",
}

ARCHIVO_AUDITADO = "feedback_curso_universitario.csv"


def _locate_raw_csvs() -> List[Path]:
    """Busca los 3 CSV originales en `data/raw/` o en la raíz del proyecto."""
    found: List[Path] = []
    for filename in RAW_CSV_FILENAMES:
        candidate_raw = RAW_DATA_DIR / filename
        candidate_root = PROJECT_ROOT / filename
        if candidate_raw.exists():
            found.append(candidate_raw)
        elif candidate_root.exists():
            found.append(candidate_root)
    return found


def _compute_file_sha256(path: Path) -> str:
    """Calcula la huella criptográfica SHA-256 de un archivo en disco."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _clean_real_csvs(paths: List[Path]) -> pd.DataFrame:
    """
    Aplica exactamente las transformaciones del Paso 1 y Paso 2 de
    `Parcial_SI_corregido.ipynb` sobre los CSV reales del parcial.
    """
    dfs = []
    for path in paths:
        temp_df = pd.read_csv(path)
        temp_df["origen_archivo"] = path.name
        dfs.append(temp_df)

    df_raw = pd.concat(dfs, ignore_index=True)

    # 1. Eliminar cabeceras repetidas infiltradas (id == 'id')
    filtro_cabecera = df_raw["id"].astype(str).str.lower().str.strip() == "id"
    df = df_raw[~filtro_cabecera].copy()

    # 2. Normalizar tipografía de cursos
    mapeo_cursos = {"Sistemas opertativos": "Sistemas operativos"}
    df["curso"] = df["curso"].replace(mapeo_cursos).astype(str).str.strip()

    # 3. Canonizar polaridad_basica
    df["polaridad_final"] = df["polaridad_basica"].apply(corregir_polaridad)

    # 4. Parsear aspectos estructurados
    if "aspectos" in df.columns:
        df["aspectos_estructurados"] = df["aspectos"].apply(parsear_aspectos)
    else:
        df["aspectos_estructurados"] = [[] for _ in range(len(df))]

    # 5. Normalización léxica y filtrado de vacíos/duplicados
    df["feedback"] = df["feedback"].astype(str).str.strip()
    df["feedback_limpio"] = df["feedback"].apply(normalizar_texto)
    df = (
        df[df["feedback_limpio"] != ""]
        .drop_duplicates(subset=["feedback_limpio"])
        .reset_index(drop=True)
    )

    # 6. Aplicar guía de desempate LLM usando llave compuesta (origen_archivo, id)
    df["id_num"] = pd.to_numeric(df["id"], errors="coerce")
    for id_caso, nueva_etiqueta in CORRECCIONES_LLM_ARCHIVO_1.items():
        mascara = (df["id_num"] == id_caso) & (df["origen_archivo"] == ARCHIVO_AUDITADO)
        if mascara.sum() == 1:
            df.loc[mascara, "polaridad_final"] = nueva_etiqueta
    df.drop(columns=["id_num"], inplace=True)

    # 7. Marcar comentarios contrastantes
    df["es_contrastante"] = df["feedback_limpio"].apply(detectar_contraste)
    return df


def _generate_synthetic_parcial_replica(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Genera una réplica demostrativa realista y determinista (n=618) que reproduce
    el esquema, las 7 asignaturas, los 4 semestres, la distribución exacta de clases
    (258 neutral, 205 negativo, 155 positivo), la alta proporción de cláusulas
    contrastantes (~76%) y los 5 casos auditados de la Tabla 1 del informe.
    """
    rng = np.random.default_rng(seed)

    # Casos reales documentados en la Tabla 1 y Tabla 5 del Informe_Parcial_SisInt.pdf
    audited_seed_rows = [
        {
            "id": 3,
            "curso": "Sistemas operativos",
            "semestre": "2024-1",
            "origen_archivo": "feedback_curso_universitario.csv",
            "feedback": (
                "El contenido fue bastante interesante, aunque algunas partes pudieron "
                "haber sido más detalladas para facilitar comprensión."
            ),
            "polaridad_final": "positivo",
            "aspectos": json.dumps(
                [
                    {"aspecto": "contenidos", "sentimiento": "positivo"},
                    {"aspecto": "material_clase", "sentimiento": "neutral"},
                ],
                ensure_ascii=False,
            ),
        },
        {
            "id": 5,
            "curso": "Arquitectura del computador",
            "semestre": "2024-1",
            "origen_archivo": "feedback_curso_universitario.csv",
            "feedback": (
                "El horario fue justo, no me dejó demasiado tiempo libre. Los parciales "
                "fueron desafiantes pero gratificantes."
            ),
            "polaridad_final": "positivo",
            "aspectos": json.dumps(
                [
                    {"aspecto": "organizacion_horario", "sentimiento": "neutral"},
                    {"aspecto": "evaluaciones_parciales", "sentimiento": "positivo"},
                    {"aspecto": "dificultad_ritmo", "sentimiento": "positivo"},
                ],
                ensure_ascii=False,
            ),
        },
        {
            "id": 6,
            "curso": "Sistemas Inteligentes",
            "semestre": "2023-2",
            "origen_archivo": "feedback_curso_universitario.csv",
            "feedback": (
                "En general, el curso ha sido bastante desafiante pero también fascinante. "
                "Los materiales se han presentado de manera clara y ordenada."
            ),
            "polaridad_final": "positivo",
            "aspectos": json.dumps(
                [
                    {"aspecto": "dificultad_ritmo", "sentimiento": "positivo"},
                    {"aspecto": "material_clase", "sentimiento": "positivo"},
                ],
                ensure_ascii=False,
            ),
        },
        {
            "id": 9,
            "curso": "Sistemas distribuidos",
            "semestre": "2023-2",
            "origen_archivo": "feedback_curso_universitario.csv",
            "feedback": (
                "El contenido fue amplio y útil, pero la disponibilidad de recursos en el "
                "aula digital era limitada. Me siento sorprendido por la falta de soporte."
            ),
            "polaridad_final": "negativo",
            "aspectos": json.dumps(
                [
                    {"aspecto": "contenidos", "sentimiento": "positivo"},
                    {"aspecto": "recursos_aula_digital", "sentimiento": "negativo"},
                ],
                ensure_ascii=False,
            ),
        },
        {
            "id": 11,
            "curso": "Programacion paralela",
            "semestre": "2023-1",
            "origen_archivo": "feedback_curso_universitario.csv",
            "feedback": (
                "Desafortunadamente, mi opinión general del curso no fue muy animada y "
                "tuve momentos de miedo ante la dificultad. Sin embargo, hubo un aspecto "
                "positivo: la organización del cronograma."
            ),
            "polaridad_final": "negativo",
            "aspectos": json.dumps(
                [
                    {"aspecto": "dificultad_ritmo", "sentimiento": "negativo"},
                    {"aspecto": "organizacion_horario", "sentimiento": "positivo"},
                ],
                ensure_ascii=False,
            ),
        },
    ]

    # Bancos de cláusulas realistas por aspecto y sentimiento (lenguaje universitario en español)
    clauses_pos = [
        ("contenidos", "el contenido del curso fue excelente, actualizado y muy pertinente para ingeniería"),
        ("metodologia_docente", "el profesor explica de forma clara, dinámica y con ejemplos prácticos muy útiles"),
        ("evaluaciones_parciales", "los parciales y talleres fueron justos, coherentes y alineados con lo visto en clase"),
        ("material_clase", "las diapositivas y guías de clase estuvieron completas, ordenadas y fáciles de repasar"),
        ("retroalimentacion", "la retroalimentación sobre los exámenes fue oportuna, detallada y constructiva"),
        ("organizacion_horario", "la organización del cronograma y el manejo del horario fueron impecables"),
        ("disponibilidad_docente", "el docente mostró excelente disponibilidad, respondió correos rápido y resolvió todas las dudas"),
        ("recursos_aula_digital", "los recursos del aula digital y los laboratorios funcionaron de maravilla"),
        ("dificultad_ritmo", "el ritmo de la materia fue adecuado y el nivel de exigencia resultó motivador y gratificante"),
    ]

    clauses_neg = [
        ("contenidos", "los contenidos resultaron desactualizados, dispersos y poco aplicables a la práctica"),
        ("metodologia_docente", "la metodología del docente fue aburrida, repetitiva y confusa al explicar los temas"),
        ("evaluaciones_parciales", "las preguntas de los parciales fueron confusas, demasiado largas y desconectadas de clase"),
        ("material_clase", "el material de clase y las diapositivas estaban desordenados, incompletos y sin ejemplos claros"),
        ("retroalimentacion", "la retroalimentación de las calificaciones tardó demasiado y no explicó en qué fallamos"),
        ("organizacion_horario", "la organización del curso fue caótica y el horario no alcanzó para cubrir los temas"),
        ("disponibilidad_docente", "la respuesta a los correos fue tardía y casi nunca hubo disponibilidad para resolver dudas"),
        ("recursos_aula_digital", "el aula digital tuvo fallas frecuentes y faltaron recursos en el laboratorio virtual"),
        ("dificultad_ritmo", "el ritmo fue demasiado rápido y la carga académica resultó excesiva y frustrante"),
    ]

    clauses_neu = [
        ("contenidos", "el contenido cumplió con el programa esperado aunque podría actualizarse un poco"),
        ("metodologia_docente", "la metodología de clase fue estándar y cumplió con los objetivos básicos del semestre"),
        ("evaluaciones_parciales", "los parciales tuvieron una dificultad promedio acorde con los temas vistos"),
        ("material_clase", "las diapositivas sirvieron de apoyo general, aunque algunos ejemplos podrían ampliarse"),
        ("retroalimentacion", "la retroalimentación se entregó en los plazos habituales de la facultad"),
        ("organizacion_horario", "el horario y la planeación del curso se mantuvieron estables durante el semestre"),
        ("disponibilidad_docente", "la comunicación con el profesor por correo y en clase fue normal"),
        ("recursos_aula_digital", "los recursos en el aula digital fueron suficientes para entregar las actividades"),
        ("dificultad_ritmo", "el ritmo de avance fue moderado con semanas tranquilas y otras más exigentes"),
    ]

    openers_pos = [
        "En general quedé muy satisfecho con la asignatura,",
        "Fue una experiencia súper enriquecedora,",
        "Recomiendo totalmente esta materia,",
        "Excelente semestre en este curso,",
        "Me gustó mucho la dinámica académica,",
        "Súper buen curso 😀,",
    ]
    openers_neg = [
        "Lamentablemente mi experiencia en el curso no fue buena,",
        "Me sentí bastante frustrado durante el semestre,",
        "Considero que la asignatura necesita mejoras urgentes,",
        "Desafortunadamente el curso dejó mucho que desear,",
        "Mi opinión general es negativa,",
        "Pésima experiencia en varios momentos 😞,",
    ]
    openers_neu = [
        "En términos generales el curso estuvo aceptable,",
        "Mi percepción de la asignatura es intermedia,",
        "El curso tiene aspectos rescatables y otros por ajustar,",
        "La experiencia durante el semestre fue regular,",
        "En balance general considero el curso normal,",
        "Hay puntos buenos y aspectos mejorables,",
    ]

    closers = [
        "espero que sigan fortaleciendo la asignatura.",
        "eso marcó bastante mi desempeño en el semestre.",
        "creo que esto ayuda a mejorar futuras cohortes.",
        "es un punto importante a tener en cuenta.",
        "en resumen así percibí el curso.",
        "vale la pena revisarlo para el próximo periodo.",
    ]

    connectors = ["pero", "sin embargo", "aunque", "no obstante"]

    # Distribución objetivo exacta del parcial: 258 neutral, 205 negativo, 155 positivo (total 618)
    target_counts = {
        "neutral": 258,
        "negativo": 205 - 2,   # ya hay 2 negativos en audited_seed_rows
        "positivo": 155 - 3,   # ya hay 3 positivos en audited_seed_rows
    }

    course_probs = np.array([145, 92, 85, 77, 76, 73, 70], dtype=float)
    course_probs /= course_probs.sum()
    sem_probs = np.array([171, 207, 227, 13], dtype=float)
    sem_probs /= sem_probs.sum()
    origin_choices = [
        "feedback_curso_universitario2.csv",
        "feedback_curso_universitario.csv",
        "feedback_curso_universitario3.csv",
    ]
    origin_probs = np.array([405, 193, 20], dtype=float)
    origin_probs /= origin_probs.sum()

    rows: List[dict] = list(audited_seed_rows)
    seen_clean = {normalizar_texto(r["feedback"]) for r in rows}

    idx_counter = 100
    for label, count in target_counts.items():
        generated = 0
        while generated < count:
            idx_counter += 1
            curso = str(rng.choice(COURSES_CANONICAL, p=course_probs))
            semestre = str(rng.choice(SEMESTERS_CANONICAL, p=sem_probs))
            origen = str(rng.choice(origin_choices, p=origin_probs))
            use_contrast = bool(rng.random() < 0.76)  # 76% contrastantes como en el parcial
            use_elongation = bool(rng.random() < 0.12)
            use_emoji = bool(rng.random() < 0.10)

            aspect_records = []
            if label == "positivo":
                c1_asp, c1_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                opener = openers_pos[int(rng.integers(0, len(openers_pos)))]
                aspect_records.append({"aspecto": c1_asp, "sentimiento": "positivo"})
                if use_contrast:
                    # Regla 3 de la guía LLM: concesión suave o cierre positivo tras adversativa
                    if rng.random() < 0.55:
                        c2_asp, c2_txt = clauses_neu[int(rng.integers(0, len(clauses_neu)))]
                        while c2_asp == c1_asp:
                            c2_asp, c2_txt = clauses_neu[int(rng.integers(0, len(clauses_neu)))]
                        text = f"{opener} {c1_txt}, aunque {c2_txt}."
                        aspect_records.append({"aspecto": c2_asp, "sentimiento": "neutral"})
                    else:
                        c2_asp, c2_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                        while c2_asp == c1_asp:
                            c2_asp, c2_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                        text = (
                            f"Al inicio la materia de {curso} parecía compleja, "
                            f"pero {c1_txt} y además {c2_txt}."
                        )
                        aspect_records.append({"aspecto": c2_asp, "sentimiento": "positivo"})
                else:
                    c2_asp, c2_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                    while c2_asp == c1_asp:
                        c2_asp, c2_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                    text = f"{opener} en {curso} {c1_txt} y {c2_txt}."
                    aspect_records.append({"aspecto": c2_asp, "sentimiento": "positivo"})
                if use_elongation:
                    text = text.replace("muy", "muuuy", 1).replace("excelente", "exceleenteee", 1)
                if use_emoji:
                    text += " 😊"

            elif label == "negativo":
                c1_asp, c1_txt = clauses_neg[int(rng.integers(0, len(clauses_neg)))]
                opener = openers_neg[int(rng.integers(0, len(openers_neg)))]
                aspect_records.append({"aspecto": c1_asp, "sentimiento": "negativo"})
                if use_contrast:
                    # Regla 1 & 2 de la guía LLM: elogio inicial pero queja crítica tras adversativa
                    c_pos_asp, c_pos_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                    while c_pos_asp == c1_asp:
                        c_pos_asp, c_pos_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                    conn = connectors[int(rng.integers(0, 3))]  # pero / sin embargo / aunque
                    if conn == "aunque":
                        text = f"{opener} {c1_txt}, aunque {c_pos_txt}."
                    else:
                        text = f"En {curso} {c_pos_txt}, {conn} {c1_txt}."
                    aspect_records.append({"aspecto": c_pos_asp, "sentimiento": "positivo"})
                else:
                    c2_asp, c2_txt = clauses_neg[int(rng.integers(0, len(clauses_neg)))]
                    while c2_asp == c1_asp:
                        c2_asp, c2_txt = clauses_neg[int(rng.integers(0, len(clauses_neg)))]
                    text = f"{opener} en {curso} {c1_txt} y además {c2_txt}."
                    aspect_records.append({"aspecto": c2_asp, "sentimiento": "negativo"})
                if use_elongation:
                    text = text.replace("pésimo", "peesiiimo", 1).replace("muy", "muuuy", 1)
                if use_emoji:
                    text += " 😠"

            else:  # neutral
                c1_asp, c1_txt = clauses_neu[int(rng.integers(0, len(clauses_neu)))]
                opener = openers_neu[int(rng.integers(0, len(openers_neu)))]
                aspect_records.append({"aspecto": c1_asp, "sentimiento": "neutral"})
                if use_contrast:
                    conn = connectors[int(rng.integers(0, len(connectors)))]
                    if rng.random() < 0.5:
                        c_pos_asp, c_pos_txt = clauses_pos[int(rng.integers(0, len(clauses_pos)))]
                        c_neg_asp, c_neg_txt = clauses_neg[int(rng.integers(0, len(clauses_neg)))]
                        while c_neg_asp == c_pos_asp:
                            c_neg_asp, c_neg_txt = clauses_neg[int(rng.integers(0, len(clauses_neg)))]
                        text = f"{opener} {c_pos_txt}, {conn} {c_neg_txt}."
                        aspect_records = [
                            {"aspecto": c_pos_asp, "sentimiento": "positivo"},
                            {"aspecto": c_neg_asp, "sentimiento": "negativo"},
                        ]
                    else:
                        c2_asp, c2_txt = clauses_neu[int(rng.integers(0, len(clauses_neu)))]
                        while c2_asp == c1_asp:
                            c2_asp, c2_txt = clauses_neu[int(rng.integers(0, len(clauses_neu)))]
                        text = f"{opener} en {curso} {c1_txt}, {conn} {c2_txt}."
                        aspect_records.append({"aspecto": c2_asp, "sentimiento": "neutral"})
                else:
                    c2_asp, c2_txt = clauses_neu[int(rng.integers(0, len(clauses_neu)))]
                    while c2_asp == c1_asp:
                        c2_asp, c2_txt = clauses_neu[int(rng.integers(0, len(clauses_neu)))]
                    text = f"{opener} en {curso} {c1_txt} y {c2_txt}."
                    aspect_records.append({"aspecto": c2_asp, "sentimiento": "neutral"})

            # Variación léxica adicional controlada para evitar duplicados exactos
            closer = closers[int(rng.integers(0, len(closers)))]
            if rng.random() < 0.65:
                text = f"{text} ({semestre}) {closer.capitalize()}"

            clean_candidate = normalizar_texto(text)
            if clean_candidate in seen_clean:
                continue

            seen_clean.add(clean_candidate)
            rows.append(
                {
                    "id": idx_counter,
                    "curso": curso,
                    "semestre": semestre,
                    "origen_archivo": origen,
                    "feedback": text,
                    "polaridad_final": label,
                    "aspectos": json.dumps(aspect_records, ensure_ascii=False),
                }
            )
            generated += 1

    df = pd.DataFrame(rows)
    df["feedback_limpio"] = df["feedback"].apply(normalizar_texto)
    df["aspectos_estructurados"] = df["aspectos"].apply(parsear_aspectos)
    df["es_contrastante"] = df["feedback_limpio"].apply(detectar_contraste)
    return df.reset_index(drop=True)


def build_and_version_dataset(
    dataset_version: str = "1.0.0",
    force_synthetic: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Construye, guarda y versiona el dataset consolidado listo para entrenamiento.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, object]]
        `(df_clean, dataset_manifest)` con la versión del dataset y su hash SHA-256.
    """
    raw_paths = [] if force_synthetic else _locate_raw_csvs()

    if raw_paths:
        df = _clean_real_csvs(raw_paths)
        source_mode = "real_csv"
        source_files = [p.name for p in raw_paths]
    else:
        df = _generate_synthetic_parcial_replica(seed=RANDOM_SEED)
        source_mode = "synthetic_parcial_replica"
        source_files = ["synthetic_replica_from_Informe_Parcial_SisInt.pdf"]

    output_csv = PROCESSED_DATA_DIR / "feedback_preprocesado_etiquetado.csv"
    df_to_save = df.drop(columns=["aspectos_estructurados"], errors="ignore")
    df_to_save.to_csv(output_csv, index=False, encoding="utf-8")

    dataset_sha256 = _compute_file_sha256(output_csv)
    manifest: Dict[str, object] = {
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha256,
        "source_mode": source_mode,
        "source_files": source_files,
        "processed_path": str(output_csv.relative_to(PROJECT_ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "class_distribution": {
            str(k): int(v) for k, v in df["polaridad_final"].value_counts().items()
        },
        "class_proportions": {
            str(k): round(float(v), 4)
            for k, v in df["polaridad_final"].value_counts(normalize=True).items()
        },
        "contrastive_ratio": round(float(df["es_contrastante"].mean()), 4),
        "courses": sorted(df["curso"].astype(str).unique().tolist()),
    }

    manifest_path = PROCESSED_DATA_DIR / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return df, manifest


if __name__ == "__main__":
    dataframe, info = build_and_version_dataset()
    print(json.dumps(info, ensure_ascii=False, indent=2))
