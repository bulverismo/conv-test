#include "conv_configs.h"
#include "lenet_inference.h" // Assumindo que este header define LeNet5, image, Predict, uint8
#include "inputs.h" 
#include <stdio.h>

//#define RANDOM 0

const int Q_FACTOR = 8; 

#define COUNT_TEST              10 


void load_data(unsigned char(**data)[28][28], unsigned char **label, const int count, unsigned char *labels_src, unsigned char *images_src)
{
  // Aponta para os arrays estáticos internos
  *label = (unsigned char *)(labels_src);

  const unsigned char *images_raw = (const unsigned char *)images_src; 
  *data = (unsigned char (*)[28][28])images_raw;
}

int load_lenet(LeNet5 **lenet, unsigned char *model_src)
{
    // Aponta diretamente para o array estático interno
    *lenet = (LeNet5 *)model_src;
    
    return 0;
}


bool lenet(void) {
  // Declaração dos vetores internos (agora dentro de lenet())


  unsigned char (*test_data)[28][28] = NULL;
  uint8 *test_label;
  LeNet5 *lenet = NULL;

  //load(lenet, MODEL_FILE);


  load_data(&test_data, &test_label, COUNT_TEST, mnist_labels_data, mnist_images_data);
  
  load_lenet(&lenet, lenet_model_data);


  image *mnist_images = (image*)test_data;

  // Testar uma única imagem (ou um loop para COUNT_TEST)
  for (int i = 0; i < MNIST_IMAGES_COUNT; i++)
  {
    //i = 0; // Exemplo para a primeira imagem
    printf("Calling prediction...\n");

    int predicted = Predict(lenet, mnist_images[i], 10); // 10 classes para dígitos
    printf("Img %d: real=%d pred=%d %s\n", 
          i, test_label[i], predicted, 
          (predicted == test_label[i]) ? "OK" : "ERR");

  }


  return true;
}

// ============================================================================
// Thread Principal para o Teste do Acelerador
// ============================================================================
void run() {

    printf("Starting accelerator test thread.\n");


    //call to lenet function
    printf("Calling lenet function (SW inference with random data)...\n");
    lenet();
    printf("Calling lenet function ended...\n");

}

int main(void)
{
    printf("Main function started. Kicking off accelerator test thread.\n");
    run();
    return 0;
}
