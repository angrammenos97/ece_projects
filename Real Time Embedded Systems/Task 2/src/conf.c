#include "conf.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "json.h"

json_value *parce_input(int argc, char *argv[]);

confFile *export_default()
{
    confFile *configuration = (confFile*)malloc(sizeof(confFile));
    configuration->uselog = 0;
    configuration->logfname = NULL;
    configuration->livelog = 0;
    configuration->queuesize = 10;
    configuration->producersnum = 3;
    configuration->consumersnum = 5;
    configuration->jobslength = 0;
    configuration->jobs = NULL;
    return configuration;
}

confFile *export_conf(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file.json>\n", argv[0]);
        return export_default();
    }
    json_value *value = parce_input(argc, argv);
    confFile *configuration = (confFile*)malloc(sizeof(confFile));
    //save json to struct
    json_object_entry *tmpval, *values;
    values = value->u.object.values; //enter into objects
    configuration->uselog = (values++)->value->u.integer;
    configuration->logfname = (char*)malloc((values)->value->u.string.length*sizeof(char));
    sprintf(configuration->logfname, "%s", (values++)->value->u.string.ptr);
    configuration->livelog = (values++)->value->u.integer;
    configuration->queuesize = (values++)->value->u.integer;
    configuration->producersnum = (values++)->value->u.integer;
    configuration->consumersnum = (values++)->value->u.integer;
    configuration->jobslength = values->value->u.array.length;
    tmpval = values;
    //struct _json_value **ar = values->value->u.array.values;  //enter into array    
    configuration->jobs = (jobConf*)malloc(configuration->jobslength*sizeof(jobConf));
    for (int x = 0; x < configuration->jobslength; x++) {
        value = tmpval->value->u.array.values[x];
        values = value->u.object.values;
        configuration->jobs[x].jobid = (values++)->value->u.integer;
        configuration->jobs[x].period = (values++)->value->u.integer;
        configuration->jobs[x].numofexec = (values++)->value->u.integer;
        configuration->jobs[x].startdelay = (values++)->value->u.integer;
        configuration->jobs[x].usedate = (values)->value->u.integer;
        if (configuration->jobs[x].usedate) {
            values++;
            configuration->jobs[x].year = (values++)->value->u.integer;
            configuration->jobs[x].month = (values++)->value->u.integer;
            configuration->jobs[x].date = (values++)->value->u.integer;
            configuration->jobs[x].hour = (values++)->value->u.integer;
            configuration->jobs[x].minute = (values++)->value->u.integer;
            configuration->jobs[x].second = (values)->value->u.integer;
        }
    }
    json_value_free(value);
    return configuration;
}

json_value *parce_input(int argc, char *argv[])
{
    char *filename =argv[1];
    struct stat filestatus;
    if (stat(filename, &filestatus) != 0)
    {
        fprintf(stderr, "File %s not found\n", filename);
        exit(-1);
    }
    int file_size = filestatus.st_size;
    char *file_contents = (char *)malloc(filestatus.st_size);
    if (file_contents == NULL)
    {
        fprintf(stderr, "Memory error: unable to allocate %d bytes\n", file_size);
        exit(-1);
    }

    FILE *fp= fopen(filename, "rt");
    if (fp == NULL)
    {
        fprintf(stderr, "Unable to open %s\n", filename);
        fclose(fp);
        free(file_contents);
        exit(-1);
    }
    if (fread(file_contents, file_size, 1, fp) != 1)
    {
        fprintf(stderr, "Unable t read content of %s\n", filename);
        fclose(fp);
        free(file_contents);
        exit(-1);
    }
    fclose(fp);
    json_char *json = (json_char *)file_contents;
    json_value *value = json_parse(json, file_size);
    if (value == NULL)
    {
        fprintf(stderr, "Unable to parse data\n");
        free(file_contents);
        exit(-1);
    }
    free(file_contents);
    return value;
}