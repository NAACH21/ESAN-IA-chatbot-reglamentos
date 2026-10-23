"""Pruebas sin API ni ChromaDB: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from citation_utils import (validate_citations, locate_literal_quote,
                            infer_program_scope, source_applies_to_scope)

DOC_DOCTORADO = "Reglam_de-los-Programas-Doctorales-de-la-Escuela-de-Negocios-2.pdf"
DOC_DIPLOMA = "Reglam_Gral.Program.de-Educacion-Ejecutiva (1).pdf"


def source(text, source=DOC_DOCTORADO, page=7, section="General"):
    return {"text": text, "source": source, "page": page, "section": section}


class TestCitasNormativas(unittest.TestCase):
    def test_first_case_same_sentence_with_different_prefixes(self):
        src = source("El profesor verifica la asistencia al inicio del horario de clase. "
                     "La tardanza podrá ser considerada como ausencia, a consideración del profesor. "
                     "3.7. Evaluación: se aplica el sistema de calificación.")
        raw = [
            {"fragmento_id": "F1", "extracto": "horario de clase. La tardanza podrá ser considerada como ausencia, a consideración del profesor."},
            {"fragmento_id": "F1", "extracto": "La tardanza podrá ser considerada como ausencia, a consideración del profesor."},
            {"fragmento_id": "F1", "extracto": "de clase. La tardanza podrá ser considerada como ausencia, a consideración del profesor. 3.7. Evaluación"},
        ]
        cited, stats = validate_citations(raw, [src])
        self.assertEqual(len(cited), 1)
        self.assertEqual(stats["duplicadas"], 2)
        self.assertEqual(cited[0]["documento"], DOC_DOCTORADO)
        self.assertEqual(cited[0]["pagina"], "7")

    def test_same_page_different_rules_must_both_remain(self):
        src = source(
            "Se cobrará el proporcional de sesiones asistidas considerando la última fecha de asistencia registrada. "
            "El retiro se deberá formular por escrito al Departamento de Registro, a más tardar "
            "en el último día de clases del curso en el que se matriculó.",
            DOC_DIPLOMA, 32, "4.2.2.4.2. Retiro de Diplomas y paquete de cursos")
        cited, _ = validate_citations([
            {"fragmento_id": "F1", "extracto": "Se cobrará el proporcional de sesiones asistidas considerando la última fecha de asistencia registrada"},
            {"fragmento_id": "F1", "extracto": "El retiro se deberá formular por escrito al Departamento de Registro, a más tardar en el último día de clases"},
        ], [src])
        self.assertEqual(len(cited), 2)
        self.assertEqual(cited[0]["pagina"], cited[1]["pagina"])

    def test_citation_without_exact_quote_is_rejected(self):
        sources = [source("La tardanza podrá ser considerada como ausencia, a consideración del profesor.")]
        cited, stats = validate_citations([
            {"fragmento_id": "F1", "extracto": "El alumno tiene derecho a justificar faltas por escrito."},
            {"fragmento_id": "F7", "extracto": "La tardanza podrá ser considerada como ausencia, a consideración del profesor."}
        ], sources)
        self.assertEqual(cited, [])
        self.assertEqual(stats["no_verificables"], 2)

    def test_bad_page_or_document_metadata_cannot_be_invented(self):
        src = source("La tardanza podrá ser considerada como ausencia, a consideración del profesor.")
        cited, _ = validate_citations([
            {"fragmento_id": "F1", "documento": "otro.pdf", "pagina": "300",
             "extracto": "La tardanza podrá ser considerada como ausencia, a consideración del profesor."}
        ], [src])
        self.assertEqual(len(cited), 1)
        self.assertEqual(cited[0]["documento"], DOC_DOCTORADO)
        self.assertEqual(cited[0]["pagina"], "7")

    def test_newline_accents_still_verify_quote(self):
        text = "La tardanza podrá ser\n considerada como ausencia, a consideración del profesor."
        r = locate_literal_quote("La tardanza podra ser considerada como ausencia, a consideracion del profesor.", text)
        self.assertIsNotNone(r)
        self.assertIn("\n", r[0])

    def test_repeated_across_chunks_one_citation(self):
        a = source("La tardanza podrá ser considerada como ausencia, a consideración del profesor.")
        b = source("El participante debe asistir a clases. La tardanza podrá ser considerada como ausencia, a consideración del profesor.")
        cited, stats = validate_citations([
            {"fragmento_id": "F1", "extracto": "La tardanza podrá ser considerada como ausencia, a consideración del profesor."},
            {"fragmento_id": "F2", "extracto": "La tardanza podrá ser considerada como ausencia, a consideración del profesor."},
        ], [a,b])
        self.assertEqual(len(cited), 1)
        self.assertEqual(stats["duplicadas"], 1)

    def test_scope_filter_blocks_regulation_from_wrong_program(self):
        self.assertEqual(infer_program_scope("Quiero retirarme de un curso del Diploma"), "diploma")
        self.assertEqual(infer_program_scope("¿Cómo justifico mis inasistencias?"), None)
        self.assertEqual(infer_program_scope("¿Cómo justifico mis inasistencias?", "Doctorado"), "doctorado")
        self.assertFalse(source_applies_to_scope(DOC_DOCTORADO, "diploma"))
        self.assertTrue(source_applies_to_scope(DOC_DIPLOMA, "diploma"))

    def test_real_pdf_doctoral_page_seven(self):
        try:
            from pypdf import PdfReader
        except ImportError:
            self.skipTest("pypdf no instalado")
        folder = Path(__file__).resolve().parents[1] / "DOCS"
        path = next(folder.rglob(DOC_DOCTORADO), None)
        if not path:
            self.skipTest("Reglamento PDF no incluido")
        text = PdfReader(path).pages[6].extract_text() or ""
        cites, _ = validate_citations([
            {"fragmento_id": "F1", "extracto": "La tardanza podrá ser considerada como ausencia, a consideración del profesor."}
        ], [source(text, page=7)])
        self.assertEqual(len(cites), 1)

    def test_real_pdf_diploma_page_32_two_independent_rules(self):
        try:
            from pypdf import PdfReader
        except ImportError:
            self.skipTest("pypdf no instalado")
        folder = Path(__file__).resolve().parents[1] / "DOCS"
        path = next(folder.rglob(DOC_DIPLOMA), None)
        if not path:
            self.skipTest("Reglamento PDF no incluido")
        text = PdfReader(path).pages[31].extract_text() or ""
        cites, stats = validate_citations([
            {"fragmento_id": "F1", "extracto": "Se cobrará el proporcional de sesiones asistidas considerando la última fecha de asistencia registrada"},
            {"fragmento_id": "F1", "extracto": "El retiro se deberá formular por escrito al Departamento de Registro"},
        ], [source(text, source=DOC_DIPLOMA, page=32)])
        self.assertEqual(len(cites), 2, stats)
        self.assertEqual(cites[0]["pagina"], "32")
        self.assertEqual(cites[1]["pagina"], "32")

    def test_no_citation_when_none_proposed(self):
        self.assertEqual(validate_citations([], [source("texto irrelevante")])[0], [])

    def test_legacy_citation_verified_against_document_and_page(self):
        s = source("El alumno tiene la obligación de asistir a todas las clases programadas.")
        cited, _ = validate_citations([{"documento": DOC_DOCTORADO,"pagina": "7",
                                    "extracto": "El alumno tiene la obligación de asistir a todas las clases programadas"}], [s])
        self.assertEqual(len(cited), 1)


if __name__ == "__main__":
    unittest.main()


class TestVinculacionV3(unittest.TestCase):
    def test_vincula_y_reutiliza_numero_deduplicado(self):
        from citation_utils import link_claims_to_citations
        text = ("La solicitud se presenta ante Coordinación del programa con la documentación correspondiente. "
                "El plazo de reconsideración de calificaciones es de cinco días calendario desde la notificación.")
        sources = [source(text, DOC_DIPLOMA, 28)]
        proposals = [
            {"fragmento_id": "F1", "extracto": "La solicitud se presenta ante Coordinación del programa con la documentación correspondiente."},
            {"fragmento_id": "F1", "extracto": "solicitud se presenta ante Coordinación del programa con la documentación correspondiente."},
            {"fragmento_id": "F1", "extracto": "El plazo de reconsideración de calificaciones es de cinco días calendario desde la notificación."},
        ]
        valid, stats = validate_citations(proposals, sources)
        self.assertEqual(stats["duplicadas"], 1)
        claims = link_claims_to_citations([
            {"texto": "Solicitar en Coordinación", "cita_indices": [2]},
            {"texto": "Plazo", "cita_indices": [3, 1]},
        ], proposals, valid, sources)
        self.assertEqual(claims, [
            {"texto": "Solicitar en Coordinación", "numeros": [1]},
            {"texto": "Plazo", "numeros": [2, 1]},
        ])

    def test_no_vincula_cita_invalida_o_inexistente(self):
        from citation_utils import link_claims_to_citations
        sources = [source("El plazo es de cinco días calendario desde la comunicación de la nota.")]
        proposals = [{"fragmento_id": "F1", "extracto": "El plazo es de tres días calendario."}]
        valid, _ = validate_citations(proposals, sources)
        self.assertEqual(link_claims_to_citations([
            {"texto": "Plazo de tres días", "cita_indices": [1, 99]},
        ], proposals, valid, sources), [])

    def test_dos_reglas_misma_pagina_con_numeros_distintos(self):
        from citation_utils import link_claims_to_citations
        sources = [source("La solicitud se presenta ante Coordinación del programa. "
                          "El plazo es de cinco días calendario desde la notificación.", DOC_DIPLOMA, 28)]
        proposals = [
            {"fragmento_id": "F1", "extracto": "La solicitud se presenta ante Coordinación del programa."},
            {"fragmento_id": "F1", "extracto": "El plazo es de cinco días calendario desde la notificación."},
        ]
        valid, _ = validate_citations(proposals, sources)
        self.assertEqual([r["numeros"] for r in link_claims_to_citations([
            {"texto": "Canal de solicitud", "cita_indices": [1]},
            {"texto": "Plazo", "cita_indices": [2]},
        ], proposals, valid, sources)], [[1], [2]])
