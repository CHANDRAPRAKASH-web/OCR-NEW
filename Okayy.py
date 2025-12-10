# produce structured parsed output
parsed_struct = parse_contact_fields(results)

# export json (include parsed)
if cfg['output'].get('export_json', True):
    json_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json")
    save_json({
        "image": os.path.basename(image_path),
        "results": results,
        "parsed": parsed_struct
    }, json_path)

# return both low-level and structured parsed object
return {
    "results": results,
    "parsed": parsed_struct
}
