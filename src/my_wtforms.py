from flask_wtf import FlaskForm
from wtforms import FileField, IntegerField, FloatField, StringField
from flask_wtf.file import FileAllowed, DataRequired
from wtforms.validators import NumberRange


class TrainValUpload(FlaskForm):
    train = FileField(label='train', validators=[
        DataRequired(),
        FileAllowed(upload_set=['csv'], message='CSV is required')])
    
    val = FileField(label='val', validators=[
        FileAllowed(upload_set=['csv'], message='CSV is required')])
    
    target_col = StringField(label='target_col', validators=[DataRequired()])

class SetRFParams(FlaskForm):
    n_estimators = IntegerField(label='n_estimators', validators=[
        NumberRange(min=1, message='Must be at least 1')
    ])
    
    max_depth = IntegerField(label='max_depth', validators=[
        NumberRange(min=1, message='Must be at least 1')
    ])

    subspace_size = FloatField(label='subspace_size', validators=[
        NumberRange(min=1e-10, max=1, message='Must be in range (0,1]')
    ])

class SetGBParams(FlaskForm):
    n_estimators = IntegerField(label='n_estimators', validators=[
        NumberRange(min=1, message='Must be at least 1')
    ])
    
    max_depth = IntegerField(label='max_depth', validators=[
        NumberRange(min=1, message='Must be at least 1')
    ])

    subspace_size = FloatField(label='subspace_size', validators=[
        NumberRange(min=1e-10, max=1, message='Must be in range (0,1]')
    ])

    learning_rate = FloatField(label='learning_rate', validators=[
        NumberRange(min=1e-10, message='Must be greater than 0')
    ])

class Predict(FlaskForm):
    X = FileField(label='train', validators=[
        DataRequired(),
        FileAllowed(upload_set=['csv'], message='CSV is required')])