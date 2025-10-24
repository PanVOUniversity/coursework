import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FaPhone, FaEnvelope, FaMapMarkerAlt } from 'react-icons/fa';
import toast from 'react-hot-toast';
import './styles.css';

const Contact = () => {
  const [formData, setFormData] = useState({ name: '', email: '', phone: '', message: '' });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      toast.success('Спасибо! Мы свяжемся с вами в ближайшее время.');
      setFormData({ name: '', email: '', phone: '', message: '' });
      setIsSubmitting(false);
    }, 2000);
  };

  return (
    <section id="contact" className="contact">
      <div className="container">
        <motion.div className="section-header" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <h2 className="section-title">Контакты</h2>
        </motion.div>

        <div className="contact-content">
          <motion.div className="contact-info" initial={{ opacity: 0, x: -50 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}>
            <div className="contact-item">
              <FaPhone className="contact-icon" />
              <div><h4>Телефон</h4><p>+7 (495) 123-45-67</p></div>
            </div>
            <div className="contact-item">
              <FaEnvelope className="contact-icon" />
              <div><h4>Email</h4><p>info@beautysalon.ru</p></div>
            </div>
            <div className="contact-item">
              <FaMapMarkerAlt className="contact-icon" />
              <div><h4>Адрес</h4><p>Москва, ул. Тверская, 10</p></div>
            </div>
          </motion.div>

          <motion.form className="contact-form" onSubmit={handleSubmit} initial={{ opacity: 0, x: 50 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}>
            <input type="text" placeholder="Имя *" value={formData.name} onChange={(e) => setFormData({...formData, name: e.target.value})} required />
            <input type="email" placeholder="Email *" value={formData.email} onChange={(e) => setFormData({...formData, email: e.target.value})} required />
            <input type="tel" placeholder="Телефон" value={formData.phone} onChange={(e) => setFormData({...formData, phone: e.target.value})} />
            <textarea placeholder="Сообщение *" value={formData.message} onChange={(e) => setFormData({...formData, message: e.target.value})} required rows="5" />
            <motion.button type="submit" className="btn btn-primary" disabled={isSubmitting} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              {isSubmitting ? 'Отправляется...' : 'Отправить'}
            </motion.button>
          </motion.form>
        </div>
      </div>
    </section>
  );
};

export default Contact;
