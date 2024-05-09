# BRAKE BALLS

# Training directory

La carpeta con datos de entrenamiento debe tener el siguiente formato donde cada subcarpeta debe contener el formato 1, 2, 3... dentro de images y el 1.obj en object

```file
folder/
│
├── dataset1/
│ ├── images/
│ │ ├── 1.jpg
│ │ ├── 2.jpg
│ │ ├── 3.jpg
│ │ ├── 4.jpg
│ │
│ └── object/
│ └── 1.obj
│
├── dataset2/
│ ├── images/
│ │ ├── 1.jpg
│ │ ├── 2.jpg
│ │ ├── 3.jpg
│ │ ├── 4.jpg
│ │
│ └── object/
│ └── 1.obj
│

```

Preparar el archivo json con las configuraciones para generar los datos:

```json
{
  // path of the folder with the objects to train
  "array_of_folder_with_objs": "/home/juan/Documentos/AAUniversidad/tesis/IntrA/generated/aneurysm/obj",
  // path to the folder where the training data will be setted
  "output_folder": "/home/juan/Documentos/AAUniversidad/tesis/BrakeBalls/data",
  // size of the images
  "size_for_img": 64,
  // number of images for each rotation -- Use 4
  "number_images_per_rotation": 4,
  // number of rotation for each object
  "number_rotations": 15,
  // this is the movement of the angle of the camera for each rotation
  "angle_per_rotation": 10,
  // this is the distance of the camera to the object
  "distance_or_radio": 70,
  // init angle in each rotation
  "init_azimut": 0,
  // end angle in each rotation
  "end_azimut": 90,
  // quantity of steps between init and end
  "cant_step_azimut": 9
}
```

Ejecutamos el comando desde paraview python para poder obtener las capturas, con -c se agrega el archivo json conconfiguraciones

```bash
/home/juan/Descargas/ParaView-5.12.0-MPI-Linux-Python3.10-x86_64/bin/pvpython   external/get-images-from-mesh-obj/prepare_data.py -c external/get-images-from-mesh-obj/config_preparing_data.json
```

Una vez que tenemos los datos

para entrenar usamo python.
