# at top of pipeline.py add:
from .parser import parse_contact_fields

# inside process_image, just before saving json or returning results:
parsed_struct = parse_contact_fields(results)
# include parsed in export
if cfg['output'].get('export_json', True):
    json_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json")
    save_json({"image": os.path.basename(image_path), "results": results, "parsed": parsed_struct}, json_path)

# and return both in memory:
return {"results": results, "parsed": parsed_struct}
