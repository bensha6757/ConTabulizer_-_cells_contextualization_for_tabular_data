from fine_tune_t5_pl import T5TemplateGeneration
from transformers import T5Tokenizer

if __name__ == '__main__':

    template_generator_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    model = T5TemplateGeneration.load_from_checkpoint(
        checkpoint_path='./t5_for_template_generation/2fg0k8t9/checkpoints/epoch=1-step=34.ckpt',
        t5_model_name="t5-base"
    )
    template_generator = model.model
    template_generator.save_pretrained('../checkpoints/template_generator')

    row = [
        "Ben Shapira # occupation # plastic surgeon # occupations",
        "Kevin Durant # height # 2.13 # NBA players",
        "picture3 # colors # red and blue # Tel Aviv Museum pictures",
        "ABBA # most known song # Gimmie! Gimmie! Gimmie! # Most known songs",
        "table # price # 55 dollars # furniture orders",
        "carakukly # most popular song # butterflies # music concerts",
        "Audi # mileage # 68050 # cars found in harry's garage",
        "vanilla # like # 12 # ice cream preferences",
        "Alon # price # 2.9 $ # buyers list"
    ]
    # row = [example.lower() for example in row]

    source_encoding = template_generator_tokenizer.batch_encode_plus(row,
                                                                     padding='max_length',
                                                                     return_tensors='pt',
                                                                     add_special_tokens=True)

    generation_output = template_generator.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=4,
        max_length=30,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True,
        return_dict_in_generate=True,
    )
    generated_ids = generation_output.sequences
    preds = [
        template_generator_tokenizer.decode(generated_id,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    ]

    print(preds)
