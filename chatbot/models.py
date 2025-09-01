from django.db import models

class Type(models.Model):
    id = models.AutoField(primary_key=True, db_column='Id')
    name = models.CharField(max_length=255, null=True, db_column='Name')

    class Meta:
        db_table = 'Types'

    def __str__(self):
        return self.name or ""


class Field(models.Model):
    id = models.AutoField(primary_key=True, db_column='Id')
    name = models.CharField(max_length=255, null=True, db_column='Name')
    type = models.ForeignKey(Type, on_delete=models.CASCADE, db_column='TypeId', related_name='fields')

    class Meta:
        db_table = 'Fields'

    def __str__(self):
        return self.name or ""


class Company(models.Model):
    id = models.AutoField(primary_key=True, db_column='Id')
    name = models.CharField(max_length=255, null=True, db_column='Name')
    description = models.TextField(null=True, db_column='Description')
    website_url = models.CharField(max_length=500, null=True, db_column='WebsiteURL')
    email = models.CharField(max_length=255, null=True, db_column='Email')
    phone = models.CharField(max_length=100, null=True, db_column='Phone')
    facebook_url = models.CharField(max_length=500, null=True, db_column='FacebookURL')
    address = models.CharField(max_length=500, null=True, db_column='Address')
    logo_image = models.CharField(max_length=500, null=True, db_column='LogoImage')

    # Many-to-many via CompanyFields
    fields = models.ManyToManyField('Field', through='CompanyField', related_name='companies')

    class Meta:
        db_table = 'Companies'

    def __str__(self):
        return self.name or ""


class CompanyField(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE, db_column='CompanyId')
    field = models.ForeignKey(Field, on_delete=models.CASCADE, db_column='FieldId')

    class Meta:
        db_table = 'CompanyFields'
        unique_together = ('company', 'field')
