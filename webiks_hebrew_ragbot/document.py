import json
import os

from dotenv import load_dotenv
load_dotenv("app/.env")
DEFINITIONS_FILE = os.getenv("DOCUMENT_DEFINITION_CONFIG", "example-conf.json")


class DocumentFieldDefinition:
    """
   Represents the definition of a document field.
   Attributes:
       field_name (str): The name of the field.
       required (bool): Indicates if the field is required. Default is False.
   """
    def __init__(self, field_name: str, required: bool = False):
        self.field_name = field_name
        self.required = required


class DocumentDefinitions:
    """
   Represents the definitions for a document.
   Attributes:
       saved_fields (dict[str, DocumentFieldDefinition]): A dictionary of saved fields.
        model_name (Name, optional): The name of the model. Default is None
        field_to_embed (Field, optional): The field it will embed. Default is None
       identifier (str): The identifier field.
       field_for_llm (str, optional): The field for LLM. Default is None.
   """
    def __init__(self, saved_fields: dict[str, DocumentFieldDefinition], model_name :str, field_to_embed:str,
                 identifier: str, field_for_llm: str = None, embed_context_fields: list[str] = None,
                 embed_title_weight: float = 0.0, title_field: str = "title"):
        self.saved_fields = saved_fields
        self.model_name = model_name
        self.field_to_embed = field_to_embed
        self.identifier = identifier
        self.field_for_llm = field_for_llm
        # Fields whose text is prepended to `field_to_embed` before embedding (e.g. ["title"]).
        # Empty by default, which reproduces the original behaviour exactly. Only the text fed
        # to the retrieval model changes; stored fields and the vector field name do not.
        self.embed_context_fields = list(embed_context_fields or [])
        # Weight of a separately embedded title vector fused into the stored paragraph vector:
        # stored = normalise(w * unit(title_vec) + (1 - w) * unit(text_vec)). 0 (default) keeps the
        # original single-text vector. Because cosine is linear in the query, searching the fused
        # vector equals w*cos(q,title) + (1-w)*cos(q,text) up to a per-document scale, so the
        # query path and the vector field are unchanged.
        self.embed_title_weight = float(embed_title_weight or 0.0)
        self.title_field = title_field


def initialize_definitions():
    """
   Initializes the document definitions by reading the configuration file.
   Raises:
       ValueError: If the identifier field is not one of the saved fields or if any model field is not one of the saved fields.
   Returns:
       DocumentDefinitions: The initialized document definitions.
   """
    with open(DEFINITIONS_FILE, 'r', encoding='utf-8') as f:
        definitions = json.load(f)

        saved_fields = definitions['saved_fields']
        model_name = definitions["model_name"]
        field_to_embed = definitions["field_to_embed"]
        identifier_field = definitions['identifier_field']
        field_for_llm = definitions.get('field_for_llm', None)
        embed_context_fields = definitions.get('embed_context_fields', [])
        embed_title_weight = float(definitions.get('embed_title_weight', 0.0))
        title_field = definitions.get('title_field', 'title')
        if not 0.0 <= embed_title_weight < 1.0:
            raise ValueError("embed_title_weight must be in [0, 1)")
        if embed_title_weight > 0 and title_field not in saved_fields.keys():
            raise ValueError(f"title_field {title_field!r} must be one of the saved fields when embed_title_weight > 0")
        if identifier_field not in saved_fields.keys():
            raise ValueError("identifier_field must be one of the saved fields, check the configuration file")

        if field_to_embed not in saved_fields.keys():
            raise ValueError(f"{field_to_embed} must be one of the saved fields {saved_fields.keys()}, check the configuration file")

        for field in embed_context_fields:
            if field not in saved_fields.keys():
                raise ValueError(f"embed_context_fields entry {field!r} must be one of the saved fields {list(saved_fields.keys())}, check the configuration file")

        return DocumentDefinitions(saved_fields, model_name, field_to_embed, identifier_field, field_for_llm,
                                   embed_context_fields=embed_context_fields,
                                   embed_title_weight=embed_title_weight, title_field=title_field)


definitions_singleton = None


def document_definition_factory():
    """
   document_definition_factory method to get the singleton instance of DocumentDefinitions.
   Returns:
       DocumentDefinitions: The singleton instance of document definitions.
   """
    global definitions_singleton
    if definitions_singleton is None:
        definitions_singleton = initialize_definitions()
    return definitions_singleton
