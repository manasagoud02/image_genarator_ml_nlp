def predict_step(image_paths):
  images=[]
  for image_path in image_paths:
    i_image=Image.open(image_path)
    if i_image.mode !="RGB":
      i_image=i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values=feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values=pixel_values.to(device)

  output_ids=model.generate(pixel_values, **gen_kwargs)

  preds=tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds=[pred.strip() for pred in preds]
  return preds


